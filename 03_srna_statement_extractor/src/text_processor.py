import re
from typing import List

import pandas as pd

class TextProcessor:
    """
    Provides functionalities for preprocessing texts before NER extraction, 
    including cleaning and sentence tokenization.
    """

    @staticmethod
    def preprocess_for_ner(fulltext: str) -> str:
        """
        Cleans the full text for NER by removing metadata, acknowledgements, 
        and references to prepare the text for entity extraction.

        Args:
            fulltext (str): The full text of the document.

        Returns:
            str: The preprocessed text.
        """
        preprocessed_text = re.sub(r"^(.*?)(Abstract|Introduction)", r"\2", fulltext, flags=re.S | re.I)
        preprocessed_text = re.sub(r"(Acknowledg(e|ing)?ments|(We|I) thank).*", "", preprocessed_text, flags=re.S | re.I)
        preprocessed_text = re.sub(r"(Note[s\s]|NOTE[S\s]|References|REFERENCES|Bibliography|BIBLIOGRAPHY).*$", "", preprocessed_text, flags=re.S)
        preprocessed_text = re.sub(r"Proc\..*?\d{4}", "", preprocessed_text)
        preprocessed_text = re.sub(r"Vol\..*?\d+", "", preprocessed_text)
        preprocessed_text = re.sub(r"pp\..*?\d+", "", preprocessed_text)

        # Look for the start of the references section and cut off the text there
        first_reference_match = re.search(r"\d{1,2}\.(\s\w+,(\s?\w+\.)+\s?[,&]?)+\(\d{4}\)[\s\w\.,]+", preprocessed_text)
        if first_reference_match:
            preprocessed_text = preprocessed_text[:first_reference_match.start()]

        return preprocessed_text.strip()

    @staticmethod
    def tokenize_sentences(fulltext: str) -> List[str]:
        """
        Tokenizes the full text into sentences while preserving specific patterns
        like figure references, citations, and 'ref.' mentions.

        Args:
            fulltext (str): The full text of the document.

        Returns:
            List[str]: A list of tokenized sentences.
        """
        figure_regex = re.compile(r"Fig\.[\s\w\(\),]+?\d+")
        citation_regex = re.compile(r"\(\d+\)")
        reference_regex = re.compile(r"ref\.")

        fulltext = re.sub(figure_regex, "FIGURE", fulltext)
        fulltext = re.sub(citation_regex, "CITATION", fulltext)
        fulltext = re.sub(reference_regex, "REFERENCE", fulltext)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", fulltext)

        sentences = [re.sub("FIGURE", "Fig.", sent) for sent in sentences]
        sentences = [re.sub("CITATION", "(citation)", sent) for sent in sentences]
        sentences = [re.sub("REFERENCE", "ref.", sent) for sent in sentences]

        return sentences

    def preprocess_texts(self, df: pd.DataFrame, text_column: str = 'fulltext') -> pd.DataFrame:
        """
        Cleans and tokenizes texts in a DataFrame, preparing them for NER extraction.
        This method modifies the DataFrame in place, adding a 'text_prep' column with preprocessed texts
        and expanding the DataFrame to have one sentence per row.

        Args:
            df (pd.DataFrame): The DataFrame containing texts to be processed.
            text_column (str): The column name in df where the texts are stored.

        Returns:
            pd.DataFrame: A modified DataFrame with an added 'text_prep' column containing preprocessed texts,
            expanded to have one sentence per row.
        """
        # Apply the cleaning process
        df['text_prep'] = df[text_column].apply(self.preprocess_for_ner)

        # Tokenize the cleaned texts into sentences
        sentences_series = df['text_prep'].apply(self.tokenize_sentences)

        # Expand the DataFrame to have one sentence per row
        expanded_df = pd.DataFrame({
            "pmcid": df['PMCID'].repeat(sentences_series.apply(len)),
            "text_prep": [sentence for sentences in sentences_series for sentence in sentences]
        })
        # Optionally, remove sentences that seem to be part of tables or lists
        expanded_df = expanded_df[~expanded_df['text_prep'].str.contains(r"(\d+\.?\d*\s?){5,}")]

        return expanded_df