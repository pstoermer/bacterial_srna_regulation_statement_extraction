import pandas as pd
from typing import Any, List, Tuple, Dict, Optional
from itertools import product
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError
from tqdm import tqdm
import time
import numpy as np


class WikidataMapper:
    """
    Manages the mapping of entities to their Wikidata QIDs and processes DataFrame rows
    to replace entity names with corresponding QIDs.
    """

    @staticmethod
    def generate_unique_entity_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Generates lists of unique sRNA and target gene entities from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to extract the entities from.
        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists - the first with unique sRNA entities and the second with unique target gene entities.
        """
        unique_srnas = set(np.concatenate(df['sRNA Entities'].dropna().values))
        unique_genes = set(np.concatenate(df['TargetGene Entities'].dropna().values))
        return list(unique_srnas), list(unique_genes)

    def query_entities_for_qids(self, entities: List[str], entity_type: str) -> Dict[str, str]:
        """
        Queries Wikidata for QIDs of a list of entities based on their type.

        Args:
            entities (List[str]): A list of entity names to query.
            entity_type (str): The type of entities to query ('srna' or 'gene').

        Returns:
            Dict[str, str]: A dictionary mapping entity names to their corresponding QIDs.
        """
        qid_map = {}
        for entity in tqdm(entities, desc=f"Querying Wikidata for {entity_type} entities"):
            qid = self._query_wikidata(entity, entity_type)
            if qid:
                qid_map[entity] = qid
        return qid_map

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
        for the description "bacterial strain" in the description of the entity linked in the property "foundInTaxon".

        In the event of a request failure or a response indicating too many requests (HTTP 429 error), the function waits for a specified
        delay before retrying the query. This process is repeated up to a maximum number of retries specified by the caller.

        Args:
            entity_name (str): The name of the entity to query.
            entity_type (str): The type of the entity (e.g., 'srna' or 'gene').
            retries (int): The number of retry attempts in case of query failure.
            delay (int): The delay in seconds before retrying after a failure.

        Returns:
            Optional[str]: The QID of the entity if found; otherwise, None.
        """
        for _ in range(retries):
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

    @staticmethod
    def map_entity_qids_and_filter(df: pd.DataFrame, srna_qid_map: Dict[str, str],
                                   gene_qid_map: Dict[str, str]) -> pd.DataFrame:
        """
        Adds corresponding QIDs for entity names in 'sRNA Entities' and 'TargetGene Entities' and filters out rows with unresolved QIDs.

        Args:
            srna_qid_map (Dict[str, str]): Mapping of sRNA entity names to their QIDs.
            gene_qid_map (Dict[str, str]): Mapping of target gene entity names to their QIDs.

        Returns:
            df (pd.DataFrame): DataFrame expanded with QIDs.
        """
        # Replace entity names with QIDs
        df['sRNA_QIDs'] = df['sRNA Entities'].apply(
            lambda entities: [srna_qid_map.get(entity) for entity in entities if entity in srna_qid_map])
        df['TargetGene_QIDs'] = df['TargetGene Entities'].apply(
            lambda entities: [gene_qid_map.get(entity) for entity in entities if entity in gene_qid_map])

        # Filter out rows where an entity was not found (i.e., None values in QID lists)
        df = df[df['sRNA_QIDs'].apply(lambda qids: all(qids)) & df['TargetGene_QIDs'].apply(lambda qids: all(qids))]

        return df

    @staticmethod
    def expand_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand DataFrame rows to create distinct rows for each sRNA and target gene combination.

        This method takes the current DataFrame and expands it so that each row represents a unique
        combination of sRNA and target gene derived from the lists in the 'sRNA Entities', 'sRNA_QIDs',
        'TargetGene Entities', and 'TargetGene_QIDs' columns. The original rows, which may contain lists
        of entities and their corresponding QIDs, are transformed into multiple rows where each row
        contains exactly one sRNA and one target gene along with their respective Wikidata QIDs.

        Args:
            df (pd.DataFrame): DataFrame to expand.

        Returns:
            pd.DataFrame: Expanded DataFrame.
        """
        rows = []
        for _, row in df.iterrows():
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
        return pd.DataFrame(rows)

    @staticmethod
    def map_relation_to_wikidata(relation_type: str) -> str:
        """
        Map relation types to corresponding Wikidata items.

        Args:
            relation_type (str): Name of the relation type

        Returns:
            str: Wikidata property code for relation type

        """
        relation_mapping = {
            'antisense inhibitor of': 'P3777',
            'regulates (molecular biology)': 'P128',
            'activator of': 'P3771'
        }
        return (relation_mapping[relation_type])

    @staticmethod
    def _refine_entities(row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refines entities by extracting unique identifiers, finding duplicates, applying
        specific refinement logic to potentially mark rows for deletion, and removing
        entities without a valid Wikidata QID.

        Args:
            row (Dict[str, Any]): One row from the DataFrame including entities and QIDs to be refined.

        Returns:
            Dict[str, Any]: Refined DataFrame row.
        """
        # Filter out entities without a valid Wikidata QID
        row['sRNA Entities'] = [e for e, qid in zip(row['sRNA Entities'], row['sRNA_QIDs']) if
                                qid != 'None' and qid is not None]
        row['TargetGene Entities'] = [e for e, qid in zip(row['TargetGene Entities'], row['TargetGene_QIDs']) if
                                      qid != 'None' and qid is not None]

        # Update QID lists to only include valid QIDs
        row['sRNA_QIDs'] = [qid for qid in row['sRNA_QIDs'] if qid != 'None' and qid is not None]
        row['TargetGene_QIDs'] = [qid for qid in row['TargetGene_QIDs'] if qid != 'None' and qid is not None]

        return row


