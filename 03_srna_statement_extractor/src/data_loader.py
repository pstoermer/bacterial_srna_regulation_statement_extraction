from typing import Optional

import pandas as pd


class DataLoader:
    """
    A utility class for loading textual data from various formats into a Pandas DataFrame.
    It supports loading from single text files where text is directly read into the DataFrame,
    or from bulk sources like Parquet files containing multiple records.

    Attributes:
        input_format (str): Specifies the format of the input data ('single' for individual text files or 'bulk' for Parquet files).
        input_path (str): The file path to the input data.
        pmcid (Optional[str]): PubMed Central ID for the source document, required if the input format is 'single'.
    """
    def __init__(self, input_format:str, input_path: str, pmcid:Optional[str] = None):
        """
        Initializes the DataLoader with the necessary information to load the input data.

        Args:
            input_format (str): The format of the input data ('single' or 'bulk').
            input_path (str): The path to the input data file.
            pmcid (Optional[str]): The PubMed Central ID for the source document. This is required if the input format is 'single'.
        """
        self.input_format = input_format
        self.input_path = input_path
        self.pmcid = pmcid

    def load_input(self) -> pd.DataFrame:
        """
        Loads the input data into a Pandas DataFrame based on the specified input format.
        For 'single' format, reads the content of a text file into the DataFrame.
        For 'bulk' format, loads a DataFrame from a Parquet file and filters it for necessary columns and valid texts.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data, with columns 'PMCID' and 'fulltext'.

        Raises:
            ValueError: If the input_format is not supported or the required columns are missing in the 'bulk' format.
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
                df = df[['PMCID', 'fulltext']].drop_duplicates().dropna(subset=['fulltext'])
                df = df[df.fulltext!=""]
            except KeyError as e:
                raise e("DataFrame does not contain columns 'PMCID' and 'fulltext'.")
        else:
            raise ValueError("Unsupported input format.")
        return df
