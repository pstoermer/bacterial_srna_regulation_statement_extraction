import pandas as pd
import re
from typing import List, Tuple, Dict


class EntityProcessor:
    """
    Processes entities extracted from texts, including separation of entity types,
    cleaning, and filtering based on specific criteria.
    """

    @staticmethod
    def separate_entities(df: pd.DataFrame, combined_entities_col: str) -> pd.DataFrame:
        """
        Separates combined entity predictions into 'sRNA Entities' and 'TargetGene Entities'.

        Args:
            df (pd.DataFrame): The DataFrame containing combined entity predictions.
            combined_entities_col (str): Column name in df where combined entity predictions are stored.

        Returns:
            pd.DataFrame: The DataFrame with separated entity columns.
        """
        srna_entities = [[] for _ in range(len(df))]
        targetgene_entities = [[] for _ in range(len(df))]
        for index, row in df.iterrows():
            for entity in row['combined_entities']:
                _, _, entity_name, entity_type = entity
                if 'SRNA' in entity_type:
                    srna_entities[index].append(entity_name)
                elif 'TARGETGENE' in entity_type:
                    targetgene_entities[index].append(entity_name)
            srna_entities[index] = list(set(srna_entities[index]))
            targetgene_entities[index] = list(set(targetgene_entities[index]))

        df['sRNA Entities'] = srna_entities
        df['TargetGene Entities'] = targetgene_entities
        return df

    @staticmethod
    def filter_cross_listed_entities(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out rows where 'sRNA Entities' and 'TargetGene Entities' share common entities.

        Args:
            df (pd.DataFrame): DataFrame to be filtered.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """

        def is_disjoint(row):
            """
            Args:
                row (pd.Series): A row of the DataFrame.

            Returns:
                bool: True if 'sRNA Entities' and 'TargetGene Entities' do not share entities, False otherwise.
            """
            srna_set = set(row['sRNA Entities'])
            gene_set = set(row['TargetGene Entities'])
            return srna_set.isdisjoint(gene_set)

        return df[df.apply(is_disjoint, axis=1)]

    @staticmethod
    def clean_entities(entities_list: List[Tuple]) -> List[Tuple]:
        """
        Cleans a list of entities by removing short entities and those matching specific patterns.

        Args:
            entities_list (List[Tuple]): A list of entity tuples.

        Returns:
            List[Tuple]: A cleaned list of entity tuples.
        """
        cleaned_list = [entity for entity in entities_list if
                        len(entity) > 2 and not re.search(r'(ABSTRACT|CITATION|FIGURE[A-Z]?|TABLE[A-Z]?)', entity)]
        return cleaned_list

    @staticmethod
    def filter_empty_entities(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out rows where 'sRNA Entities' or 'TargetGene Entities' are empty after processing.

        Args:
            df (pd.DataFrame): DataFrame to be filtered.

        Returns:
            pd.DataFrame: Filtered DataFrame with non-empty entity columns.
        """
        return df[
            (df['sRNA Entities'].apply(lambda x: len(x) > 0)) &
            (df['TargetGene Entities'].apply(lambda x: len(x) > 0))
            ].reset_index(drop=True)