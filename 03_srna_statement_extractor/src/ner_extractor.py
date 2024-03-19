import json
import torch
import spacy
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch.nn.functional as F
from torch.nn.functional import softmax
from typing import Dict, List, Tuple
from tqdm import tqdm


class NERExtractor:
    """
    Extracts named entities from texts using both BioBERT and SciSpaCy models.
    """

    def __init__(self, biobert_model_path: str, scispacy_model_path: str, id_to_label_path: str):
        """
        Initializes the NER extractor with the specified BioBERT and SciSpaCy models.

        Args:
            biobert_model_path (str): Path to the pretrained BioBERT model.
            scispacy_model_path (str): Path to the SciSpaCy model.
            id_to_label_path (str): Path to the JSON file mapping model output IDs to labels.
        """
        self.bert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_path)
        self.bert_model = AutoModelForTokenClassification.from_pretrained(biobert_model_path)
        self.nlp = spacy.load(scispacy_model_path)
        self.id_to_label = {int(k): v for k, v in self._load_json(id_to_label_path).items()}
        self.bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def _load_json(path: str) -> Dict:
        """
        Loads a JSON file and returns its content.

        Args:
            path (str): Path to the JSON file

        Returns:
            dict: Content of the JSON file
        """

        with open(path, 'r') as file:
            return json.load(file)

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
                    current_entity[0] + text[current_entity[2]:token_end], current_entity[1], token_end,
                    current_entity[3])

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
                        current_entity[0] + text[token_start:token_end], current_entity[1], token_end,
                        current_entity[3])
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
    def _preprocess_and_normalize(predictions: List[Tuple[str, int, int, str]]):
        """
        Preprocess and normalize predictions to flatten them into the desired format.
        Each entity or token is represented as a tuple: (text, start, end, tag).

        Args:
            predictions (List[Tuple[str, int, int, str]]): The original predictions in any format that includes BIO tags or similar.

        Returns:
            normalized_docs (List[Tuple[str, int, int, str]]): A list of documents, where each document is a list of tuples (text, start, end, tag).
        """

        normalized_doc = []
        for token, start, end, tag in predictions:
            # Normalize the tag by removing BIO prefixes if present
            normalized_tag = tag[2:] if tag.startswith("B-") or tag.startswith("I-") else tag
            normalized_doc.append((token, start, end, normalized_tag))

        return normalized_doc

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
        biobert_preds = self._preprocess_and_normalize(biobert_preds)
        # Convert normalized predictions to dictionaries for easier comparison
        biobert_dict = {(start, end): (text, label) for text, start, end, label in biobert_preds}
        scispacy_dict = {(start, end): (text, label) for text, start, end, label in scispacy_preds}

        merged_predictions = []

        # Process sciSpaCy predictions first to prioritize them
        for span, scispacy_pred in scispacy_dict.items():
            merged_predictions.append((*span, *scispacy_pred))

        # Process BioBERT predictions that are not in sciSpaCy
        for span, biobert_pred in biobert_dict.items():
            if span not in scispacy_dict:
                merged_predictions.append((*span, *biobert_pred))

        return merged_predictions

    def extract_entities(self, text: str) -> List[Tuple]:
        """
        Extracts entities from a text, using both BioBERT and SciSpaCy models, and merges the predictions.

        Args:
            texts (List[str]): The text to process.

        Returns:
            A list that contains tuples for the entities predicted in a single text.
        """

        biobert_preds = self._biobert_predict(text)
        scispacy_preds = self._scispacy_predict(text)

        return self.merge_predictions_with_priority(biobert_preds, scispacy_preds)



