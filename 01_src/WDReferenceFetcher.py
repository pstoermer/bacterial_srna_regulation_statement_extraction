import urllib.parse

import pandas as pd
from tqdm import tqdm
from wikidataintegrator.wdi_core import WDItemEngine


class WDReferencesFetcher:
    """Fetches references for sRNA interactions from Wikidata based on QID."""
    def __init__(self, QID):
        """
        Initialize the fetcher with a specific QID.
        
        Args:
            QID (str): The QID for which to fetch data from Wikidata.
        """
        self.QID = QID

    def get_wd_label(self):
        """Fetches the label for the specified QID from Wikidata."""
        with open('Label_Fetch_Query.rq', 'r') as query_file:
            query_template = query_file.read()
            
        QUERY = query_template.replace("#QID#", self.QID)
        results = WDItemEngine.execute_sparql_query(QUERY)['results']['bindings']
        
        if len(results) == 0:
            print("Query returns no items for the specified Q-ID.")
            return ""
        elif len(results) == 1:
            return results[0]['label']['value']
        else:
            print("Query returns more than one item for the same Q-ID.")
            return ""

    def get_interacted_RNA_references(self):
        """Fetches references for interacted RNA from Wikidata based on QID."""
        interacted_RNA_references = []
        row_nums = 0
        
        with open('../02_queries/ALL_INTERACTED_SRNA_QUERY.rq', 'r') as query_file:
            query_template = query_file.read()
            
        QUERY = query_template.replace("#QID#", self.QID)
        results = WDItemEngine.execute_sparql_query(QUERY)['results']['bindings']

        if len(results) == 0:
            return None

        for result in results:
            row_nums += 1
            tmp_quote = urllib.parse.quote(result['quote']['value'])
            pmc_url = f"/Article_Viewer.html?pmcid=PMC{result['PMCID']['value']}&quote={tmp_quote}"
            pmc_url = pmc_url.replace('.', '%2E').replace('-', '%2D')
            
            # Constructing the row with available data
            row = [row_nums,
                   result.get('rnaLabel', {}).get('value', ''),
                   result.get('altLabel', {}).get('value', ''),
                   result.get('propLabel', {}).get('value', ''),
                   result.get('targetLabel', {}).get('value', ''),
                   result.get('quote', {}).get('value', ''),
                   pmc_url,
                   result.get('PMCID', {}).get('value', ''),
                   result.get('rna', {}).get('value', '')]
            interacted_RNA_references.append(row)

        data_tbl_cols = ['#', 'sRNA', 'sRNA synonyms', 'Type of Regulation', 'Target Gene', 'Quote', 'Quote from', 'PMCID', 'Wikidata']
        return pd.DataFrame(interacted_RNA_references, columns=data_tbl_cols)

def main():
    """Main function to fetch and compile RNA interaction data from Wikidata."""
    wikidata_df = pd.read_csv('/data/01_raw_data/query.csv')
    wikidata_df['q_ids'] = wikidata_df.rna.apply(lambda x: str(x).split('/')[-1])
    wikidata_df = wikidata_df[~wikidata_df.q_ids.isnull()]
    q_ids = list(set(wikidata_df['q_ids']))

    
    data_tbl_cols = ['#', 'sRNA', 'sRNA synonyms', 'Type of Regulation', 'Target Gene', 'Quote', 'Quote from', 'PMCID', 'Wikidata']
    reference_df = pd.DataFrame(columns=data_tbl_cols)
    
    for q_id in tqdm(q_ids):
        fetched_df = WDReferencesFetcher(q_id).get_interacted_RNA_references()
        if isinstance(fetched_df, pd.DataFrame):
            reference_df = pd.concat([reference_df, fetched_df], ignore_index=True)
    
    reference_df.drop_duplicates(inplace=True)

    reference_df.to_csv('../rna_labels_with_references.csv', index=False)

if __name__ == '__main__':
    main()