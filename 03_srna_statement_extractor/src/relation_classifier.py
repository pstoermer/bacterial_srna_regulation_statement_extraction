import os
import joblib
import re

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RelationClassifier:
    """
    A classifier for identifying the relationship expressed in a given piece of text
    using a pretrained sequence classification model.

    Attributes:
        model (AutoModelForSequenceClassification): A pretrained model for sequence classification.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the sequence classification model.
        label_encoder (LabelEncoder): An encoder for transforming labels to integers and vice versa.
        model_path (str): Path to the directory containing the pretrained model and label encoder.
    """

    def __init__(self, model_path: str):
        """
        Initializes the RelationClassifier with a pretrained model and tokenizer.

        Args:
            model_path (str): Path to the directory containing the pretrained model, tokenizer,
                              and label encoder joblib file.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.joblib'))
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def classify(self, text: str) -> str:
        """
        Classifies the type of relation expressed in the given text.

        The method applies a predefined confidence threshold to determine if a prediction is reliable.
        It also uses a simple regex pattern to quickly identify specific relations (e.g., activation relations)
        without the need for model inference, as a form of rule-based optimization.

        Args:

        """
        confidence_threshold = 0.6
        regex_pattern = r"activat(es?|ion|or|ing)"

        if re.search(regex_pattern, text):
            return 'activator of'

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
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

        return predicted_label_name