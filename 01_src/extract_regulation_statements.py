import argparse
import time
from datetime import datetime as dt
import json
import os

from itertools import product

import re
from typing import Any, Dict, List, Optional, Tuple

import requests
from nltk.corpus import wordnet

from urllib.error import HTTPError
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.functional import softmax

from tqdm import tqdm

tqdm.pandas()
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
import uuid

# MODEL PATHS
RELATION_CLASSIFIER_PATH = "/data/03_models/relation_classifier/"
SCISPACY_MODEL_PATH = "/data/03_models/finetuned/scispacy/finetuned_scispacy_ner/"
BIOBERT_MODEL_PATH = "/data/03_models/finetuned/biobert/finetuned_biobert_ner/"
ID_TO_LABEL_PATH = "/data/02_training_data/id_to_label.json"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments with input_format, input_path, pmcid, and output_path.
    """
    parser = argparse.ArgumentParser(description='Extract sRNA regulation statements.')
    parser.add_argument('--input_format',
                        type=str,
                        required=True,
                        choices=['single', 'bulk'],
                        help='Type of input to process (single txt file or bulk process a parquet file containing a pd.DataFrame')
    parser.add_argument('--input_path',
                        type=str,
                        required=True,
                        help='Path to the input file.')
    parser.add_argument('--pmcid',
                        type=str,
                        help='PubMed Central ID for the source document (required if input format is "single").')
    parser.add_argument('--output_path',
                        type=str,
                        required=True,
                        help='Path to save the RDF output.')
    return parser.parse_args()


class DataLoader:
    """
    Loads input data from specified paths.
    """

    def __init__(self, input_format: str, input_path: str, pmcid: Optional[str] = None):
        self.input_format = input_format
        self.input_path = input_path
        self.pmcid = pmcid

    def load_input(self) -> pd.DataFrame:
        """
        Load input data based on the specified format.

        Returns:
            pd.DataFrame: The loaded and formatted DataFrame.
        """
        if self.input_format == 'single':
            if not self.pmcid:
                raise ValueError("PMCID is required for text input format.")
            with open(self.input_path, 'r') as file:
                text = file.read()
            df = pd.DataFrame({'PMCID': [self.pmcid], 'fulltext': [text]})
        elif self.input_format == 'bulk':
            df = pd.read_parquet(self.input_path)
            try:
                df = df[['PMCID', 'fulltext']]
                # Remove entries with no fulltext
                df = df.drop_duplicates()
                df = df[(~df.fulltext.isnull()) & (df.fulltext != "")]
            except KeyError as e:
                raise e("DataFrame does not contain columns 'PMCID' and 'fulltext'.")
        else:
            raise ValueError("Unsupported input format.")
        print(f'Found a total of {df.shape[0]} fulltext(s).')
        return df


class StatementExtractor:
    """
    Extracts regulation statements from scientific texts, leveraging NER models and enriching with Wikidata.
    """

    def __init__(self, df: pd.DataFrame, relation_classifier_path: str,
                 biobert_model_name: str, scispacy_model_name: str, id_to_label_path: str):
        """
        Initializes the StatementExtractor.

        Args:
            df (pd.DataFrame): DataFrame containing the texts to process.
            relation_classifier_path (str): Path to the relation classification model.
            biobert_model_name (str): Name of the BioBERT model.
            scispacy_model_name (str): Name of the SciSpaCy model.
            id_to_label_path (str): Path to the mapping from model output IDs to labels.
        """
        self.df = df
        self.relation_classifier = AutoModelForSequenceClassification.from_pretrained(relation_classifier_path)
        self.label_encoder = joblib.load(os.path.join(relation_classifier_path, 'label_encoder.joblib'))
        self.relation_tokenizer = AutoTokenizer.from_pretrained(relation_classifier_path)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
        self.bert_model = AutoModelForTokenClassification.from_pretrained(biobert_model_name)
        self.nlp = spacy.load(scispacy_model_name)
        self.id_to_label = {int(k): v for k, v in self._load_json(id_to_label_path).items()}
        self.bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_json(path: str) -> Dict:
        """
        Loads a JSON file and returns its content.

        Args:
            path (str): Path to the JSON file.

        Returns:
            Dict: Content of the JSON file.
        """
        import json
        with open(path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _preprocess_for_ner(fulltext: str) -> str:
        """
        Preprocess the full text for NER by removing metadata, acknowledgements, and references.

        Args:
        - fulltext (str): The full text of the document.

        Returns:
        - str: The preprocessed text.
        """
        # Remove titles, authors, journals, and dates
        preprocessed_text = re.sub(r"^(.*?)(Abstract|Introduction)", r"\2", fulltext, flags=re.S | re.I)

        # Remove acknowledgements
        preprocessed_text = re.sub(r"(Acknowledg(e|ing)?ments|(We|I) thank).*", "", preprocessed_text,
                                   flags=re.S | re.I)

        # Remove notes and referencesN
        preprocessed_text = re.sub(r"(Note[s\s]|NOTE[S\s]|References|REFERENCES|Bibliography|BIBLIOGRAPHY).*$", "",
                                   preprocessed_text, flags=re.S)

        # Additional cleanup for any lingering headers or footers
        preprocessed_text = re.sub(r"Proc\..*?\d{4}", "", preprocessed_text)
        preprocessed_text = re.sub(r"Vol\..*?\d+", "", preprocessed_text)
        preprocessed_text = re.sub(r"pp\..*?\d+", "", preprocessed_text)

        # In case references are simply listed below the actual text without a heading, find the first reference and remove everything from there on
        first_reference_match = re.search(r"\d{1,2}\.(\s\w+,(\s?\w+\.)+\s?[,&]?)+\(\d{4}\)[\s\w\.,]+",
                                          preprocessed_text)
        if first_reference_match:
            # Remove everything from the start of the first reference to the end of the text
            preprocessed_text = preprocessed_text[:first_reference_match.start()]

        return preprocessed_text.strip()

    @staticmethod
    def _tokenize_sentences(fulltext: str) -> str:
        """
        Tokenize full texts into sentences, preserving specific patterns.

        Args:
            fulltext (str): The full text of the document.

        Returns:
            List[str]: List of tokenized sentences.
        """

        # Define regular expression patterns to preserve specific patterns
        figure_pattern = r"Fig\.[\s\w\(\),]+?\d+"  # Pattern to match figure mentions
        citation_pattern = r"\(\d+\)"  # Pattern to match citations
        reference_pattern = r"ref\."  # Pattern to match "ref."

        # Compile the regular expression patterns
        figure_regex = re.compile(figure_pattern)
        citation_regex = re.compile(citation_pattern)
        reference_regex = re.compile(reference_pattern)

        # Replace figure mentions, citations, and "ref." with placeholders to preserve them
        fulltext = re.sub(figure_regex, r"FIGURE", fulltext)
        fulltext = re.sub(citation_regex, r"CITATION", fulltext)
        fulltext = re.sub(reference_regex, r"REFERENCE", fulltext)

        # Use the existing sentence tokenizer after preserving specific patterns
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", fulltext)

        # Restore preserved patterns in each sentence
        sentences = [re.sub(r"FIGURE", lambda m: m.group(0), sent) for sent in sentences]
        sentences = [re.sub(r"CITATION", lambda m: m.group(0), sent) for sent in sentences]
        sentences = [re.sub(r"REFERENCE", lambda m: m.group(0), sent) for sent in sentences]

        return sentences

    def preprocess_texts(self):
        """
        Preprocesses text for Named Entity Recognition (NER) by removing certain sections.

        Removes metadata, acknowledgements, and references from the full text.
        Tokenizes the preprocessed text into sentences.

        """
        self.df['text_prep'] = self.df['fulltext'].apply(self._preprocess_for_ner)

        sentences_series = self.df['text_prep'].apply(self._tokenize_sentences)
        self.df = pd.DataFrame({
            "pmcid": self.df['PMCID'].repeat(sentences_series.apply(len)),
            "text_prep": [sentence for sentences in sentences_series for sentence in sentences]
        })
        # Remove sentences with long lists of numbers (with or without decimal points)
        # This pattern usually indicates a table of some sort.
        self.df[~self.df['text_prep'].str.contains(r"(\d+\.?\d*\s?){5,}")]

    def classify_relations(self):
        # Prepare the model and move it to the appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.relation_classifier.to(device)

        confidence_threshold = 0.6
        regex_pattern = r"activat(es?|ion|or|ing)"
        predictions = []

        # Iterate over rows in the dataframe
        for text in tqdm(self.df['text_prep']):
            # Check if the regex pattern for 'activator of' matches the text
            if re.search(regex_pattern, text):
                predictions.append('activator of')
                continue

            # Tokenize text
            encoded_input = self.relation_tokenizer(text, return_tensors='pt', padding=True, truncation=True,
                                                    max_length=512)
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

            # Predict
            with torch.no_grad():
                outputs = self.relation_classifier(**encoded_input)
            logits = outputs.logits

            # Convert logits to probabilities and get the predicted class
            probabilities = softmax(logits, dim=1)
            max_probs, preds = torch.max(probabilities, dim=1)

            prediction_np = preds.cpu().numpy()[0]
            max_prob_np = max_probs.cpu().numpy()[0]

            if max_prob_np <= confidence_threshold:
                predicted_label_name = "none"
            else:
                # Get the label name from the encoder
                predicted_label_name = self.label_encoder.inverse_transform([prediction_np])[0]

            predictions.append(predicted_label_name)

        # Update the dataframe with predictions
        self.df['relation'] = predictions

    def _biobert_predict(self, text: str) -> List[Tuple]:
        """
        Predict entities in the text using BioBERT model, employing a sliding window approach if necessary.

        Args:
            text (str): The text to predict entities from.

        Returns:
            List[Tuple]: List of predicted entities as tuples (text, start, end, label).
        """
        tokens = self.bert_tokenizer.tokenize(text)
        if len(tokens) + 2 <= 512:  # Using the default max_length of 512 tokens
            return self._process_text(text)
        else:
            return self._process_text_sliding_window(text)

    def _process_text(self, text: str) -> List[Tuple[str, int, int, str]]:
        """
        Processes the entire text through the BioBERT model without applying the sliding window technique.

        Args:
            text (str): The text to process.

        Returns:
            List[Tuple]: List of predicted entities as tuples (text, start, end, label).
        """
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                                     return_offsets_mapping=True)
        offset_mapping = inputs.pop('offset_mapping').squeeze().tolist()

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.bert_model.device) for k, v in inputs.items()}

        # Perform prediction
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_tag_ids = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]

        predictions, current_entity = [], None
        for idx, tag_id in enumerate(predicted_tag_ids):
            if idx >= len(offset_mapping):  # Ensure index is within offset_mapping bounds
                break
            label = self.id_to_label[tag_id]
            token_start, token_end = offset_mapping[idx]

            if label.startswith('B-') or (label.startswith('I-') and current_entity is None):
                if current_entity is not None:
                    predictions.append(current_entity)
                current_entity = (text[token_start:token_end], token_start, token_end, label)

            elif label.startswith('I-') and current_entity is not None:
                current_entity = (
                current_entity[0] + text[current_entity[2]:token_end], current_entity[1], token_end, current_entity[3])

            else:
                if current_entity is not None:
                    predictions.append(current_entity)
                    current_entity = None

        if current_entity is not None:
            predictions.append(current_entity)

        return predictions

    def _process_text_sliding_window(self, text: str, overlap: int = 50) -> List[Tuple[str, int, int, str]]:
        """
        Processes text through a model using a sliding window approach, considering special tokens.

        Args:
            text (str): The text to process.
            overlap (int): The overlap between windows.

        Returns:
            List[Tuple]: List of predicted entities as tuples (text, start, end, label).
        """

        predictions = []
        # Adjust window size to account for [CLS] and [SEP]
        window_size = 512 - 2  # Reserving space for [CLS] and [SEP]
        step_size = window_size - overlap

        # Tokenize text
        tokenized_input = self.bert_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        full_tokens = tokenized_input.tokens()
        offset_mapping = tokenized_input['offset_mapping']
        num_tokens = len(full_tokens)

        for start_idx in range(0, num_tokens, step_size):
            end_idx = min(start_idx + window_size, num_tokens)

            # Prepare segment tokens with [CLS] and [SEP] as needed by the model
            segment_tokens = full_tokens[start_idx:end_idx]
            segment_ids = self.bert_tokenizer.convert_tokens_to_ids(segment_tokens)
            # Add [CLS] and [SEP] token ids
            input_ids = [self.bert_tokenizer.cls_token_id] + segment_ids + [self.bert_tokenizer.sep_token_id]
            segment_offset_mapping = [(0, 0)] + offset_mapping[start_idx:end_idx] + [(0, 0)]

            # Convert ids to tensors and prepare input dictionary
            inputs = {
                "input_ids": torch.tensor([input_ids], dtype=torch.long).to(self.bert_model.device),
                "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long).to(self.bert_model.device)
            }

            # Perform prediction
            with torch.no_grad():
                outputs = self.bert_model(**inputs)

            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predicted_tag_ids = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]

            current_entity = None
            for idx, tag_id in enumerate(predicted_tag_ids[1:-1], start=1):  # Exclude special tokens
                label = self.id_to_label[tag_id.item()]
                token_start, token_end = segment_offset_mapping[idx]

                if label.startswith('B-') or (label.startswith('I-') and current_entity is None):
                    if current_entity is not None:
                        predictions.append(current_entity)
                    current_entity = (text[token_start:token_end], token_start, token_end, label)
                elif label.startswith('I-') and current_entity is not None:
                    current_entity = (
                    current_entity[0] + text[token_start:token_end], current_entity[1], token_end, current_entity[3])
                else:
                    if current_entity is not None:
                        predictions.append(current_entity)
                        current_entity = None
            if current_entity is not None:
                predictions.append(current_entity)

        if not predictions:
            return []

        # Sort predictions by start index
        predictions.sort(key=lambda x: x[1])

        consolidated = []
        current_entity = predictions[0]

        for next_entity in predictions[1:]:
            _, current_end, _, current_label = current_entity
            next_text, next_start, next_end, next_label = next_entity

            # Check if next entity overlaps or is adjacent with the same label
            if next_start <= current_end and next_label == current_label:
                # Merge entities
                merged_text = current_entity[0] + ' ' + next_text
                merged_entity = (merged_text.strip(), current_entity[1], next_end, current_label)
                current_entity = merged_entity
            else:
                # No overlap or different label, add current entity to consolidated list
                consolidated.append(current_entity)
                current_entity = next_entity

        # Add the last entity
        consolidated.append(current_entity)

        # Remove duplicates
        predictions = list(set(consolidated))

        # Return consolidated predictions
        return predictions

    def _scispacy_predict(self, text: str) -> List[Tuple]:
        """
        Processes text using the SciSpaCy Named Entity Recognition (NER) pipeline to extract predictions.

        Args:
            text (str): The text to process.

        Returns:
            List[Tuple]: List of predicted entities as tuples (text, start, end, label).
        """
        # Process the text using the SciSpacy NER pipeline
        doc = self.nlp(text)

        # Extract NER predictions from the doc
        predictions = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        return predictions

    @staticmethod
    def _is_actual_word(entity_text: str) -> bool:
        """
        Determines if a given text is an actual English word based on WordNet entries.

        Args:
            entity_text (str): The text to be checked.

        Returns:
            bool: True if the text is an actual word, False otherwise.
        """
        # Normalize the entity_text by converting it to lowercase
        entity_text = entity_text.lower()
        # Check if the text has an entry in WordNet
        return bool(wordnet.synsets(entity_text))

    @staticmethod
    def _preprocess_and_normalize(predictions):
        """
        Preprocess and normalize predictions to flatten them into the desired format.
        Each entity or token is represented as a tuple: (text, start, end, tag).

        Parameters:
        - predictions: The original predictions in any format that includes BIO tags or similar.

        Returns:
        - A list of documents, where each document is a list of tuples (text, start, end, tag).
        """
        normalized_docs = []

        for doc in predictions:
            normalized_doc = []
            for token, start, end, tag in doc:
                # Normalize the tag by removing BIO prefixes if present
                normalized_tag = tag[2:] if tag.startswith("B-") or tag.startswith("I-") else tag
                normalized_doc.append((token, start, end, normalized_tag))
            normalized_docs.append(normalized_doc)

        return normalized_docs

    def merge_predictions_with_priority(self, biobert_preds: List[Tuple[int, int, str, str]],
                                        scispacy_preds: List[Tuple[int, int, str, str]]) -> List[
        Tuple[int, int, str, str]]:
        """
        Merges predictions from different models with priority given to certain predictions.

        Args:
            biobert_preds (List[Tuple[int, int, str, str]]): Predictions from the BioBERT model.
            scispacy_preds (List[Tuple[int, int, str, str]]): Predictions from the SciSpaCy model.

        Returns:
            List[Tuple[int, int, str, str]]: Merged list of predictions.
        """
        merged_predictions = []
        biobert_preds = self._preprocess_and_normalize(biobert_preds)
        for bioBERT_doc, sciSpaCy_doc in tqdm(zip(biobert_preds, scispacy_preds)):

            # Convert normalized predictions to dictionaries for easier comparison
            bioBERT_dict = {(start, end): (text, label) for text, start, end, label in bioBERT_doc}
            sciSpaCy_dict = {(start, end): (text, label) for text, start, end, label in sciSpaCy_doc}

            doc_merged_preds = []
            # Process sciSpaCy predictions first to prioritize them
            for span, sciSpaCy_pred in sciSpaCy_dict.items():
                doc_merged_preds.append((*span, *sciSpaCy_pred))

            # Process BioBERT predictions that are not in sciSpaCy
            for span, bioBERT_pred in bioBERT_dict.items():
                if span not in sciSpaCy_dict:
                    doc_merged_preds.append((*span, *bioBERT_pred))

            merged_predictions.append(doc_merged_preds)
        return merged_predictions

    def separate_entities(self):
        """
        Separates entities into 'sRNA Entities' and 'TargetGene Entities' based on their type.

        Modifies:
        - self.df: The DataFrame containing the data with entities separated into 'sRNA Entities'
          and 'TargetGene Entities'.
        """
        srna_entities = [[] for _ in range(len(self.df))]
        targetgene_entities = [[] for _ in range(len(self.df))]
        for index, row in self.df.iterrows():
            for entity in row['combined_entities']:
                _, _, entity_name, entity_type = entity
                if 'SRNA' in entity_type:
                    srna_entities[index].append(entity_name)
                elif 'TARGETGENE' in entity_type:
                    targetgene_entities[index].append(entity_name)
            srna_entities[index] = list(set(srna_entities[index]))
            targetgene_entities[index] = list(set(targetgene_entities[index]))

        self.df['sRNA Entities'] = srna_entities
        self.df['TargetGene Entities'] = targetgene_entities

    @staticmethod
    def filter_cross_listed_entities(row) -> bool:
        """
        Filters rows where 'sRNA Entities' and 'TargetGene Entities' share common entities.

        Parameters:
        - row (pd.Series): A row of the DataFrame.

        Returns:
        bool: True if 'sRNA Entities' and 'TargetGene Entities' do not share entities, False otherwise.
        """
        # Convert both columns to sets for easy comparison
        srna_set = set(row['sRNA Entities'])
        gene_set = set(row['TargetGene Entities'])
        # Check if intersection is empty (meaning no entities are shared)
        return srna_set.isdisjoint(gene_set)

    @staticmethod
    def clean_entities(entities_list):
        List[str]

    ) -> List[str]:
    """
    Cleans a list of entities by removing short entities and those matching specific patterns.

    Parameters:
    - entities_list (List[str]): A list of entity names.

    Returns:
    List[str]: A cleaned list of entities.
    """
    cleaned_list = [entity for entity in entities_list if
                    len(entity) > 2 and not re.search(r'(ABSTRACT|CITATION|FIGURE[A-Z]?|TABLE[A-Z]?)', entity)]
    return cleaned_list

    @staticmethod
    def _refine_entities(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Refines entities by extracting unique identifiers, finding duplicates, applying
        specific refinement logic to potentially mark rows for deletion, and removing
        entities without a valid Wikidata QID.
        """
        # Your existing logic for refining entities

        # Additionally, filter out entities without a valid Wikidata QID
        row['sRNA Entities'] = [e for e, qid in zip(row['sRNA Entities'], row['sRNA_QIDs']) if
                                qid != 'None' and qid is not None]
        row['TargetGene Entities'] = [e for e, qid in zip(row['TargetGene Entities'], row['TargetGene_QIDs']) if
                                      qid != 'None' and qid is not None]

        # Update QID lists to only include valid QIDs
        row['sRNA_QIDs'] = [qid for qid in row['sRNA_QIDs'] if qid != 'None' and qid is not None]
        row['TargetGene_QIDs'] = [qid for qid in row['TargetGene_QIDs'] if qid != 'None' and qid is not None]

        return row

    def generate_unique_entity_lists(self) -> Tuple[List[str], List[str]]:
        """
        Generates lists of unique sRNA and target gene entities from the DataFrame.

        Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists - the first with unique sRNA entities and the second with unique target gene entities.
        """
        unique_srnas = set(np.concatenate(self.df['sRNA Entities'].dropna().values))
        unique_genes = set(np.concatenate(self.df['TargetGene Entities'].dropna().values))
        return list(unique_srnas), list(unique_genes)

    def _query_entity_for_qid(self, entity: tuple, entity_type: str) -> Optional[str]:
        """
        Queries for the QID of a given entity based on its name and type.

        Parameters:
        - entity (tuple): The entity tuple, assumed to contain the entity name as its third element.
        - entity_type (str): The type of the entity (e.g., 'srna' or 'gene').

        Returns:
        - Optional[str]: The QID of the entity if found; otherwise, None.
        """
        entity_name = entity  # Assuming the entity name is the third element in the tuple
        qid = self._query_wikidata(entity_name, entity_type)
        return qid

    def query_entities_for_qids(self, entities: List[str], entity_type: str) -> Dict[str, str]:
        """
       Queries Wikidata for QIDs of a list of entities based on their type.

       Parameters:
       - entities (List[str]): A list of entity names to query.
       - entity_type (str): The type of entities to query ('srna' or 'gene').

       Returns:
       Dict[str, str]: A dictionary mapping entity names to their corresponding QIDs.
       """

    qid_map = {}
    for entity in tqdm(entities):
        qid = self._query_wikidata(entity, entity_type)  # Your existing query function
        if qid:
            qid_map[entity] = qid
    return qid_map


def _process_entities(self, row: dict) -> pd.Series:
    """
    Processes entities related to sRNA and target genes from a given row of data.

    Parameters:
    - row (dict): A dictionary containing 'sRNA Entities' and 'TargetGene Entities' as keys.

    Returns:
    - pd.Series: A pandas Series object with two elements: lists of sRNA and target gene QIDs,
      indexed by 'sRNA_QID' and 'targetGene_QID'.
    """
    # Process 'sRNA Entities'
    srna_qids = [self._query_entity_for_qid(entity, 'srna') for entity in row['sRNA Entities']]

    # Process 'targetGene Entities'
    gene_qids = [self._query_entity_for_qid(entity, 'gene') for entity in row['TargetGene Entities']]

    return pd.Series([srna_qids, gene_qids])


@staticmethod
def _query_wikidata(entity_name: str, entity_type: str, retries: int = 5, delay: int = 5) -> Optional[str]:
    """
    Performs a query against Wikidata to retrieve the QID (Wikidata item ID) associated with a given entity name and type.
    This function constructs SPARQL queries tailored to the entity type (e.g., sRNA or gene) and attempts to match the entity
    name with items in Wikidata. It employs a retry mechanism to handle transient network errors or query rate limits.

    The function differentiates between entity types to construct specific queries that optimize the chances of accurately
    matching the entity name with the correct Wikidata item. It filters the results to include only those items that are relevant
    to the specified entity type, such as ensuring that sRNA entities are matched with non-coding RNA items in Wikidata and that
    gene entities are associated with protein-coding genes. Further it checks, that entities belong to a bacteria stem, by checking
    for the description "bacterial stem" in the description of the entity linked in the property "foundInTaxon".

    In the event of a request failure or a response indicating too many requests (HTTP 429 error), the function waits for a specified
    delay before retrying the query. This process is repeated up to a maximum number of retries specified by the caller.

    Parameters:
    - entity_name (str): The name of the entity to query.
    - entity_type (str): The type of the entity (e.g., 'srna' or 'gene').
    - retries (int): The number of retry attempts in case of query failure.
    - delay (int): The delay in seconds before retrying after a failure.

    Returns:
    - Optional[str]: The QID of the entity if found; otherwise, None.
    """
    for attempt in range(retries):
        try:
            endpoint_url = "https://query.wikidata.org/sparql"
            sparql = SPARQLWrapper(endpoint_url)

            if 'srna' in entity_type.lower():
                query = f"""SELECT * WHERE {{
                      SERVICE wikibase:mwapi {{
                          bd:serviceParam wikibase:api "EntitySearch" .
                          bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                          bd:serviceParam mwapi:search "{entity_name.lower()}" .
                          bd:serviceParam mwapi:language "en" .
                          ?item wikibase:apiOutputItem mwapi:item .
                          ?num wikibase:apiOrdinal true .
                      }}
                      ?item (wdt:P279|wdt:P31) ?type .
                      ?item schema:description ?description .
                      ?item wdt:P703 ?foundInTaxon.
                      ?foundInTaxon schema:description ?foundInTaxonDescription FILTER(LANG(?foundInTaxonDescription) = "en").
                      FILTER(LANG(?description) = "en" && CONTAINS(LCASE(?description), "non-coding rna"))
                      FILTER(CONTAINS(LCASE(?foundInTaxonDescription), "bacterial strain"))
                    }} ORDER BY ASC(?num) LIMIT 1"""
            elif 'gene' in entity_type.lower():
                query = f"""SELECT * WHERE {{
                      SERVICE wikibase:mwapi {{
                          bd:serviceParam wikibase:api "EntitySearch" .
                          bd:serviceParam wikibase:endpoint "www.wikidata.org" .
                          bd:serviceParam mwapi:search "{entity_name.lower()}" .
                          bd:serviceParam mwapi:language "en" .
                          ?item wikibase:apiOutputItem mwapi:item .
                          ?num wikibase:apiOrdinal true .
                      }}
                      ?item (wdt:P279|wdt:P31) ?type .
                      ?item schema:description ?description .
                      ?item wdt:P703 ?foundInTaxon.
                      ?foundInTaxon schema:description ?foundInTaxonDescription FILTER(LANG(?foundInTaxonDescription) = "en").
                      FILTER(LANG(?description) = "en" && CONTAINS(LCASE(?description), "protein"))
                      FILTER(CONTAINS(LCASE(?foundInTaxonDescription), "bacterial strain"))
                    }} ORDER BY ASC(?num) LIMIT 1"""
            else:
                return "Invalid entity type specified."

            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)

            results = sparql.query().convert()
            if results['results']['bindings']:
                time.sleep(1)
                return results['results']['bindings'][0]['item']['value'].split('/')[-1]
            else:
                return None
        except HTTPError as e:
            if e.code == 429:
                print(f"HTTP Error 429: Too Many Requests. Retrying in {delay} seconds...")
                time.sleep(delay)

            else:
                print(f"HTTP error: {e}")
                return None
        except Exception as e:
            print(f"Error querying Wikidata: {e}")
            return None
    print("Maximum retries reached. Returning None")
    return None


def map_entities_to_wikidata(self):
    """
    Wrapper function to enrich entities in the DataFrame with Wikidata QIDs.
    Iterates over each row, queries Wikidata for QIDs based on entity names and types,
    and updates the DataFrame with the obtained QIDs.
    """
    self.df[['sRNA_QIDs', 'TargetGene_QIDs']] = self.df.progress_apply(self._process_entities, axis=1)


def map_entity_qids_and_filter(self, srna_qid_map: Dict[str, str], gene_qid_map: Dict[str, str]):
    """
    Replaces entity names in 'sRNA Entities' and 'TargetGene Entities' with their corresponding QIDs and filters out rows with unresolved QIDs.

    Parameters:
    - srna_qid_map (Dict[str, str]): Mapping of sRNA entity names to their QIDs.
    - gene_qid_map (Dict[str, str]): Mapping of target gene entity names to their QIDs.
    """
    # Replace entity names with QIDs
    self.df['sRNA_QIDs'] = self.df['sRNA Entities'].apply(
        lambda entities: [srna_qid_map.get(entity) for entity in entities if entity in srna_qid_map])
    self.df['TargetGene_QIDs'] = self.df['TargetGene Entities'].apply(
        lambda entities: [gene_qid_map.get(entity) for entity in entities if entity in gene_qid_map])

    # Filter out rows where an entity was not found (i.e., None values in QID lists)
    self.df = self.df[
        self.df['sRNA_QIDs'].apply(lambda qids: all(qids)) & self.df['TargetGene_QIDs'].apply(lambda qids: all(qids))]


def expand_rows(self):
    """
    Expand DataFrame rows to create distinct rows for each sRNA and target gene combination.

    This method takes the current DataFrame and expands it so that each row represents a unique
    combination of sRNA and target gene derived from the lists in the 'sRNA Entities', 'sRNA_QIDs',
    'TargetGene Entities', and 'TargetGene_QIDs' columns. The original rows, which may contain lists
    of entities and their corresponding QIDs, are transformed into multiple rows where each row
    contains exactly one sRNA and one target gene along with their respective Wikidata QIDs.
    """
    rows = []
    for _, row in self.df.iterrows():
        # Create all combinations of sRNA and target gene for the row
        combinations = product(zip(row['sRNA Entities'], row['sRNA_QIDs']),
                               zip(row['TargetGene Entities'], row['TargetGene_QIDs']))

        for (sRNA, sRNA_QID), (targetGene, targetGene_QID) in combinations:
            new_row = row.to_dict()
            new_row['sRNA Entities'] = sRNA
            new_row['TargetGene Entities'] = targetGene
            new_row['sRNA_QIDs'] = sRNA_QID
            new_row['TargetGene_QIDs'] = targetGene_QID
            rows.append(new_row)

    self.df = pd.DataFrame(rows)


def process(self):
    """
    Orchestrate the full regulation statement processing pipeline.

    This method coordinates the sequence of processing steps applied to regulation statements,
    including text preprocessing, relation classification, entity prediction using BioBERT and SciSpacy,
    post-processing of predictions, merging predictions, mapping entities and relations to Wikidata, and
    expanding rows for unique sRNA - target gene combinations.
    """
    self.preprocess_texts()
    print(f'Found a total of {self.df.shape[0]} unique sentences.')
    self.classify_relations()
    self.df = self.df[self.df['relation'] != 'none']
    print(f'Found the following relations:')
    print(self.df.relation.value_counts())

    print('Extracting NER...')
    # This is a simplified example; you might need to adapt it based on your specific needs and model interfaces
    self.df['biobert_entities'] = self.df['text_prep'].progress_apply(self._biobert_predict)
    self.df['scispacy_entities'] = self.df['text_prep'].progress_apply(self._scispacy_predict)

    print('Combining Model predictions...')
    self.df['combined_entities'] = self.merge_predictions_with_priority(self.df['biobert_entities'].to_list(),
                                                                        self.df['scispacy_entities'].to_list())
    self.df = self.df[self.df['combined_entities'].apply(lambda x: len(x) > 0)]

    self.df = self.df.reset_index()
    self.separate_entities()
    self.df = self.df[
        (self.df['sRNA Entities'].apply(lambda x: len(x) > 0)) & (self.df['TargetGene Entities'].apply(lambda x:
                                                                                                       len(x) > 0))].reset_index(
        drop=True)
    self.df = self.df[self.df.apply(self.filter_cross_listed_entities, axis=1)]
    self.df['sRNA Entities'] = self.df['sRNA Entities'].apply(self.clean_entities)
    self.df['TargetGene Entities'] = self.df['TargetGene Entities'].apply(self.clean_entities)
    self.df = self.df[
        (self.df['sRNA Entities'].apply(lambda x: len(x) > 0)) & (self.df['TargetGene Entities'].apply(lambda x:
                                                                                                       len(x) > 0))].reset_index(
        drop=True)

    print('Performing Quality Check with Wikidata.')
    unique_srnas, unique_genes = self.generate_unique_entity_lists()
    srna_qid_map = self.query_entities_for_qids(unique_srnas, 'srna')
    gene_qid_map = self.query_entities_for_qids(unique_genes, 'gene')
    self.map_entity_qids_and_filter(srna_qid_map, gene_qid_map)
    self.map_relation_to_wikidata()

    self.df = self.df.apply(self._refine_entities, axis=1).dropna().reset_index(drop=True)
    self.df = self.df[(self.df['sRNA Entities'].apply(lambda x: len(x) > 0)) & (
        self.df['TargetGene Entities'].apply(lambda x: len(x) > 0))].reset_index(drop=True)
    self.expand_rows()


class RDFCreator:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the RDFCreator with a DataFrame and sets up namespaces and the RDF graph.

        Args:
            df (pd.DataFrame): The DataFrame containing regulation statements and entity information.

        Attributes:
            df (pd.DataFrame): Stores the input DataFrame.
            WD, P, EX, PMC (Namespace): RDFLib Namespaces for Wikidata entities, properties, example resources, and PMC articles.
            g (Graph): An RDFLib Graph object for storing RDF triples.
        """

        self.df = df
        self.WD = Namespace("http://www.wikidata.org/entity/")
        self.P = Namespace("http://www.wikidata.org/prop/direct/")
        self.EX = Namespace("http://example.org/resource/")
        self.PMC = Namespace("https://www.ncbi.nlm.nih.gov/pmc/articles/")
        self.g = Graph()
        self.g.bind("wd", self.WD)
        self.g.bind("p", self.P)
        self.g.bind("ex", self.EX)
        self.g.bind("pmc", self.PMC)

    def create_rdf(self):
        """
        Iterates over the DataFrame rows and creates RDF triples representing each regulation statement,
        linking sRNAs, target genes, types of regulation, PMCIDs, and text descriptions.
        """
        for index, row in self.df.iterrows():
            # Generate a unique identifier for each statement to ensure URI uniqueness
            statement_uri = URIRef(f"{self.EX}{str(uuid.uuid4())[:8]}")

            # Create URIs for sRNA, target gene, and the regulation relation type
            srna_uri = URIRef(self.WD[row['sRNA_QIDs']])
            targetgene_uri = URIRef(self.WD[row['TargetGene_QIDs']])
            relation_uri = URIRef(row["relation_prop"])

            # Add triples for the regulation relationship, regulated entity, regulator, and target gene
            self.g.add((statement_uri, RDF.type, self.EX.RegulationStatement))
            self.g.add((statement_uri, self.P[row["relation_prop"]], targetgene_uri))
            self.g.add((statement_uri, self.EX.hasRegulator, srna_uri))
            self.g.add((statement_uri, self.EX.hasTargetGene, targetgene_uri))

            # Optionally add the source PMCID and descriptive text as triples, if available
            if "pmcid" in row and row["pmcid"]:
                pmcid_uri = URIRef(self.PMC[row["pmcid"]])
                self.g.add((statement_uri, self.EX.hasSource, pmcid_uri))
            if "text_prep" in row and row["text_prep"]:
                self.g.add((statement_uri, RDFS.comment, Literal(row["text_prep"])))


def main(args):
    """
    Main function to orchestrate the processing of input data into RDF triples.

    This function loads input data, applies a statement extraction process to identify and classify
    regulation statements involving sRNAs and target genes, and then converts these statements
    and their associated metadata into RDF format, which is subsequently written to a file.

    Args:
        args (argparse.Namespace): A namespace object containing command-line arguments. Expected
        arguments include 'input_format' (format of the input data), 'input_path' (path to the input
        data file), 'pmcid' (PubMed Central ID for linking the source document), and 'output_path'
        (path where the RDF output will be saved).

    Returns:
        None: This function does not return a value but writes RDF data to a file specified by
        'args.output_path'.

    Raises:
        Various exceptions can be raised depending on the implementation of the DataLoader,
        StatementExtractor, and RDFCreator classes. For example, FileNotFoundError could be raised
        if 'args.input_path' does not point to a valid file.
    """
    # Load data using the specified input format and path
    data_loader = DataLoader(args.input_format, args.input_path, args.pmcid)
    df = data_loader.load_input()

    # Initialize a StatementExtractor object with the loaded DataFrame and model paths
    extractor = StatementExtractor(df, RELATION_CLASSIFIER_PATH, BIOBERT_MODEL_PATH, SCISPACY_MODEL_PATH,
                                   ID_TO_LABEL_PATH)

    # Process the data to extract and classify regulation statements
    extractor.process()
    if not extractor.df.empty:
        # If the DataFrame is not empty, i.e., regulation statements were found and processed
        # Initialize RDFCreator with the processed DataFrame
        rdf_creator = RDFCreator(extractor.df)
        # Generate RDF triples from the processed data
        rdf_creator.create_rdf()

        # Write the RDF data to the specified output file in XML format
        print(f'Writing results to file: {args.output_path}')
        with open(args.output_path, "wb") as f:
            f.write(rdf_creator.g.serialize(format="xml").encode())

    # If no regulation statements were found in the input data
    else:
        print(f'No regulation statements found.')


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    main(args)










