import os
from typing import List, Dict, Tuple, Any
from itertools import chain
import re 
import pickle
import json
import spacy 
from spacy.training import Example
from spacy.tokens import Doc
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def read_texts(file_path:str)->List[str]:
    """
    Read text data from a file and return it as a list of strings.

    Parameters:
    - file_path (str): The path to the input file.

    Returns:
    - List[str]: A list of strings containing the text data.
    """
    with open(file_path) as file:
        texts = [line.strip() for line in file.readlines()]
    return texts


def read_entity_dict(file_path:str)->Dict[str, List[str]]:
    """
    Read an entity dictionary from a JSON file, transform values to lowercase,
    deduplicate, and sort entities for each type.

    Parameters:
    - file_path (str): The path to the JSON file containing the entity dictionary.

    Returns:
    - dict: A dictionary with entity types as keys and lists of entities as values.
    """
    with open(file_path) as file:
        entity_dict = json.load(file)
    # Transform entity values to lowercase, deduplicate, and sort entities for each type
    for key, values in entity_dict.items():
        # Convert each value to lowercase
        lower_values = [value.lower() for value in values]
        # Deduplicate and sort the lowercase values
        entity_dict[key] = sorted(set(lower_values))
    return entity_dict

def load_data(file_path: str) -> Any:
    """
    Loads data from a pickle file.

    Args:
        file_path (str): The path to the pickle file from which data is to be loaded.

    Returns:
        Any: The data deserialized from the pickle file. 

    Raises:
        FileNotFoundError: If no file exists at the specified `file_path`.

    Note:
        - This function uses Python's built-in `pickle` module for deserialization. The `pickle` module can
          serialize and deserialize a wide variety of Python objects. 
        - This function is designed to work with binary pickle files, as indicated by opening the file in "rb" mode.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    with open(file_path, "rb") as file:
        return pickle.load(file)



def combine_texts_and_entities(texts:List[str], entity_dict:Dict[str, List[str]])->List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Combine input texts with their assigned NER tags into a list of tuples.

    Parameters:
    - texts (List[str]): A list of input texts.
    - entity_dict (Dict[str, List[str]]): A dictionary with entity types as keys and lists of entities as values.

    Returns:
    - List[Tuple[str, List[Tuple[int,int,str]]]]: A list of tuples containing text and its associated NER tags.
    """
    custom_ner_tags_with_spans = [assign_custom_ner_tags_with_spans(text, entity_dict) for text in texts]
    data = list(zip(texts, custom_ner_tags_with_spans))
    return data
    


def evaluate_spacy(ner_model: spacy.language.Language, docs: List[spacy.tokens.Doc]) -> Dict[str, float]:
    """
    Evaluate a spaCy Named Entity Recognition (NER) model on a list of documents.

    Args:
    ner_model (spaCy NER model): The spaCy NER model to be evaluated.
    docs (list of spaCy Doc objects): A list of spaCy Doc objects containing the text
                                     and true entity annotations.

    Returns:
    evaluation_results (dict): A dictionary containing evaluation metrics, including
                               precision, recall, and F1-score for the NER model on
                               the given documents.
    """
    examples = []
    for doc in docs:
        # Create an Example object for each Doc
        examples.append(Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}))
    
    # Evaluate the model
    return ner_model.evaluate(examples)

    
def extract_tokens_and_labels(doc_list: List[Doc]) -> Tuple[List[str], List[str]]:
    """
    Extracts tokens and their associated named entity labels from a list of spaCy Doc objects.

    Args:
    doc_list (List[Doc]): A list of spaCy Doc objects containing named entities.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing two lists:
        - tokens (List[str]): A list of token texts extracted from the documents.
        - labels (List[str]): A list of named entity labels corresponding to each token.
                              If a token does not have a named entity label, it is assigned the label 'O' (Outside).

    Example:
    >>> doc_list = [spacy_doc1, spacy_doc2, ...]  # List of spaCy Doc objects
    >>> tokens, labels = extract_tokens_and_labels(doc_list)
    >>> print(tokens)
    ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
    >>> print(labels)
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    """
    tokens = []
    labels = []

    # Iterate through the list of spaCy Doc objects
    for doc in doc_list:
        # Iterate through tokens in the document
        for token in doc:
            # Append the token text to the tokens list
            tokens.append(token.text)
            
            # Append the named entity label if available, or 'O' if not
            labels.append(token.ent_type_ if token.ent_type_ else 'O')

    return tokens, labels


def spacy_to_biobert_input(docs, tokenizer, label_to_id):
    """
    Converts SpaCy docs to BioBERT input format with IOB tagging.

    Args:
    - docs: A list of SpaCy Doc objects.
    - tokenizer: A tokenizer instance compatible with the BioBERT model.
    - label_to_id: A dictionary mapping BIO labels to integer IDs.

    Returns:
    - A list of dictionaries, each representing tokenized inputs with corresponding labels.
    """
    biobert_data = []

    for doc in docs:
        text = doc.text
        tokenized_input = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            is_split_into_words=False
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])

        # Initialize labels array with -100 for all tokens initially
        labels = [-100] * len(tokens)

        # Reset labels for non-special tokens to 'O' (outside entity)
        for idx, token in enumerate(tokens):
            if token not in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                labels[idx] = label_to_id['O']

        # Iterate over entities in the SpaCy Doc
        for ent in doc.ents:
            ent_start, ent_end = ent.start_char, ent.end_char
            ent_label = ent.label_

            # Iterate over token offsets to find matching entities
            for idx, (start, end) in enumerate(tokenized_input['offset_mapping']):
                # Skip special tokens
                if start is None and end is None:
                    continue
                
                # Check if token is within entity boundaries
                if start >= ent_start and end <= ent_end:
                    if start == ent_start:
                        label = f"B-{ent_label}"
                    else:
                        label = f"I-{ent_label}"
                    
                    labels[idx] = label_to_id.get(label, label_to_id['O'])

        # Ensure PAD tokens remain labeled as -100
        labels = [-100 if token == tokenizer.pad_token else label for token, label in zip(tokens, labels)]

        biobert_data.append({
            "input_ids": tokenized_input['input_ids'],
            "attention_mask": tokenized_input['attention_mask'],
            "labels": labels
        })
    
    return biobert_data


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, id_to_label: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Aligns prediction indices with their corresponding labels and filters out ignored predictions.

    Args:
        predictions (np.ndarray): The raw predictions from the model, shaped as (batch_size, sequence_length, num_labels).
        label_ids (np.ndarray): The true label indices for each token, shaped as (batch_size, sequence_length).
        id_to_label (Dict[int, str]): A dictionary mapping label indices to their string representation.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: A tuple containing two lists:
            - The first list contains the predicted labels for each token in each sample of the batch.
            - The second list contains the true labels for each token in each sample of the batch.
            Both lists are structured as a list of lists, where each sublist corresponds to a sample in the batch.
    Note:
        - The function assumes that `id_to_label` is a dictionary mapping label
          indices to their corresponding string representations.
        - It is important that the `predictions` array contains logits and not
          softmax probabilities. The function internally applies argmax to find
          the most likely label index for each token.
        - The special value -100 in `label_ids` is used to signify tokens that
          should not be considered for evaluation (e.g., padding tokens or special
          tokens depending on the model's tokenization process).
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != -100:  # Ignore special tokens
                labels_list[i].append(id_to_label[label_ids[i][j]])
                preds_list[i].append(id_to_label[preds[i][j]])

    return preds_list, labels_list


def compute_ner_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes NER metrics including precision, recall, and F1 score based on the provided predictions and references.

    Args:
        predictions (List[List[str]]): A list of lists where each sublist contains predicted labels for a sequence.
        references (List[List[str]]): A list of lists where each sublist contains true labels for a sequence.

    Returns:
        Dict[str, float]: A dictionary containing the computed metrics:
            - 'precision': The precision of the predictions.
            - 'recall': The recall of the predictions.
            - 'f1': The F1 score of the predictions.
            - 'classification_report': A detailed classification report as a string.

    Note:
        - The `seqeval` library is used to compute the metrics, which is specifically designed for sequence labeling
          tasks like named entity recognition. It handles BIO (Begin, Inside, Outside) tagging schemes and calculates
          metrics in a way that considers the entire span of named entities.
        - This function assumes that both `predictions` and `references` are aligned, i.e., they have the same
          structure and length, with corresponding elements representing the labels for the same token.
    """
    precision = precision_score(references, predictions, average='weighted')
    recall = recall_score(references, predictions, average='weighted')
    f1 = f1_score(references, predictions, average='weighted')
    report = classification_report(references, predictions)

    return {"precision": precision, "recall": recall, "f1": f1, "classification_report": report}