import string
import re 
import requests
import json
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import metapub

from Bio import Entrez
import xml.etree.ElementTree as ET

import spacy
import textract

def append_gene_data(df:pd.DataFrame, quote: str, genes: List[str], rows_to_add: List[dict], srna_names: Optional[List[str]] = None) -> List[dict]:
    """
    Appends modified row data for specific genes to a list.

    Args:
        data_df (DataFrame): The DataFrame containing the data.
        quote_index (int): The index of the quote to filter the data.
        genes (list): List of gene names to be appended.
        rows_to_add (list): List to which the modified rows are appended.
        srna_names (list, optional): List of sRNA names to be used. If None, the original sRNA name is retained.
    """
    # Iterate through each gene specified in the genes list
    for gene in genes:
        if srna_names:
            # If specific sRNA names are provided, iterate through each sRNA name
            for srna in srna_names or [None]:
                # Filter the DataFrame for the row matching the specified quote and select the last matching row
                row = df[df.text_prep == quote].iloc[-1]
                # Convert the selected row to a dictionary
                row_dict = row.to_dict()
                # Modify the dictionary to include the current gene and sRNA name
                row_dict['gene_name_mentioned'] = gene
                row_dict['srna_name_mentioned'] = srna
                rows_to_add.append(row_dict)
        else:
            # If no specific sRNA names are provided, proceed with the original sRNA name
            row = df[df.text_prep == quote].iloc[-1]
            row_dict = row.to_dict()
            row_dict['gene_name_mentioned'] = gene
            rows_to_add.append(row_dict)
    return rows_to_add

def assign_custom_ner_tags_with_spans(text:str, entity_dict:Dict[str, List[str]])->List[Tuple[int, int, str]]:
    """
    Assign custom Named Entity Recognition (NER) tags to entities found in the text.

    Parameters:
    - text (str): The input text.
    - entity_dict (Dict[str, List[str]]): A dictionary with entity types as keys and lists of entities as values.

    Returns:
    - List[List[int,int,str]]: A list of tuples containing the start and end character positions and the entity type.
    """
    entities = []
    sorted_entities = sorted([(entity, label) for label, words in entity_dict.items() for entity in words], 
                             key=lambda x: len(x[0]), reverse=True)
    
    for entity, label in sorted_entities:
        start = text.lower().find(entity.lower())
        while start != -1:
            end = start + len(entity)
            # Ensure the entity doesn't start or end in the middle of another word
            if (start == 0 or not text[start-1].isalnum()) and (end == len(text) or not text[end].isalnum()):
                entities.append((start, end, label))
            start = text.find(entity, start + 1)
    
    return entities


def clean_fulltexts(text: bytes) -> str:
    """
    Cleans the full text extracted from a PDF.

    Args:
        text (bytes): The full text in bytes format.

    Returns:
        str: The cleaned full text as a string.
    """
    # Transform from bytes string to normal string
    text = text.decode()
    # Replace newlines by spaces
    text = text.replace('\n', ' ')
    # Replace symbols
    text = text.replace('ﬁ', "fi")
    return text


def combine_text_and_entities(text:str, srna:List[str], target_gene:List[str])->Tuple[List[str, List[Tuple[int, int, str]]]]:
    """
    Combine input texts with their assigned NER tags into a list of tuples.

    Parameters:
    - texts (List[str]): A list of input texts.
    - entity_dict (Dict[str, List[str]]): A dictionary with entity types as keys and lists of entities as values.

    Returns:
    - List[Tuple[str, List[Tuple[int,int,str]]]]: A list of tuples containing text and its associated NER tags.
    """
    entity_dict ={'SRNA': srna, 'TARGETGENE': target_gene}
    custom_ner_tags_with_spans = assign_custom_ner_tags_with_spans(text, entity_dict)
    data = tuple([text, custom_ner_tags_with_spans])
    return data

def create_dataset(data:List[Tuple[str, List[Tuple[int,int,str]]]], spacy_model)->List[spacy.tokens.Doc]:
    """
    Create a dataset with Named Entity Recognition (NER) tags based on the provided data.

    Parameters:
    - data (List[Tuple[str, List[Tuple[int,int,str]]]]): A list of tuples containing text and its associated NER tags.

    Returns:
    - List[spacy.tokens.Doc]: A list of spaCy Doc objects with NER tags.
    """
    dataset = []
    for text, annotation in data:
        doc = spacy_model.make_doc(text)
        ents = []
        for start, end, label in annotation:
            span = doc.char_span(start, end, label=label)
            if span is not None:
                ents.append(span)
                
        doc.ents = ents
        dataset.append(doc)
    return dataset


# Function to find the name or synonym in the quote, case-insensitively
def find_name_in_quote(quote: str, name: str, synonyms: List[str], gene_id: Optional[str] = None) -> Union[str, List[str]]:
    """
    Searches for occurrences of a name, its synonyms, and optionally a gene ID within a given quote.

    Args:
        quote (str): The text in which to search for the name, synonyms, and gene ID.
        name (str): The primary name to search for in the quote.
        synonyms (List[str]): A list of synonyms for the name to also search for in the quote.
        gene_id (Optional[str], optional): An optional gene ID to search for in the quote. Defaults to None.

    Returns:
        Union[str, List[str]]: A single name or a list of names found in the quote. Returns "unknown" if no names are found.
    """
    name_indices = []  # Initialize a list to store indices where names are found
    names = []  # Initialize a list to store the actual names found
    quote_lower = quote.lower()  # Convert the quote to lowercase for case-insensitive searching

    # Search for the primary name in the quote
    if name.lower() in quote_lower:
        name_indices.extend([m.start() for m in re.finditer(name.lower(), quote.lower())])
    # Extract the names based on the found indices
    for i in name_indices:
        names.append(quote[i:i+len(name)])

    # If synonyms are provided, search for each synonym in the quote
    if synonyms and synonyms != ['']:
        synonyms.sort(key=len, reverse=True) # Sort synonyms by length in descending order for accurate matching
        for syn in synonyms:
            if syn.lower() in quote_lower:
                name_indices.extend([m.start() for m in re.finditer(syn.lower(), quote_lower)])
                for i in name_indices:
                    names.append(quote[i:i+len(syn)])

    # If a gene ID is provided, search for the gene ID and its underscore-stripped version in the quote
    if gene_id:
        if gene_id.lower() in quote_lower:
            name_indices.extend([m.start() for m in re.finditer(gene_id.lower(), quote_lower)])
            # Extract the names based on the new indices found for the gene ID
            for i in name_indices:
                    names.append(quote[i:i+len(gene_id)])
                         
        elif gene_id.replace('_', '').lower() in quote_lower:
            name_indices.extend([m.start() for m in re.finditer(gene_id.replace('_', '').lower(), quote_lower)])
            for i in name_indices:
                    names.append(quote[i:i+len(gene_id.replace('_', ''))])
                
    # Remove duplicate names and return the result
    names = list(set(names))
    if names:
        if len(names)==1:
            return names[0]
        return names
    return "unknown"
    
def get_additional_gene_synonyms(gene_id: str) -> Union[str, List[str]]:
    """
    Fetches additional synonyms for a given target gene from GenomeNET and NCBI

    Args:
        gene_id (str): The ID of the gene to fetch information for.

    Returns:
        Union[str, List[str]]: A string or a list of strings.
                                If successful, returns a list of synonyms
                                (also known as names) for the gene, including the symbol if available.
                                If an error occurs, returns an empty string.
    Note:
        This function requires 'requests' and 'BeautifulSoup' libraries installed
        in the Python environment. It also assumes stable HTML structures for the
        source websites.

    Example:
        >>> get_additional_gene_synonyms('b0929')
        ['ompF', 'cmlB', 'cry', 'ECK0920', 'nfxB', 'tolF']

    """
    url_genomenet = f'https://www.genome.jp/dbget-bin/www_bget?eco:{gene_id}'
    url_aureowiki = f'https://aureowiki.med.uni-greifswald.de/{gene_id}'

    # FOR IDs with b*****
    if gene_id[0] == "b":
        # Make the HTTP GET request to GenomeNET
        try:
            response = requests.get(url_genomenet)
            response.raise_for_status()
        except requests.HTTPError as e:
            raise e("Error fetching GenomeNET data.")
            return ""

        genome_soup = BeautifulSoup(response.content, 'lxml')

        # Extract the symbol
        symbol = ''
        try:
            symbol_td = genome_soup.find('th', string='Symbol').find_next_sibling('td')
        except AttributeError:
            symbol_td = None
        if symbol_td:
            symbol = symbol_td.get_text(strip=True)

        # Extract the NCBI-GeneID link
        ncbi_geneid_link = None
        for tr in genome_soup.find_all('tr'):
            # Check if any <td> or <th> in this row contains 'NCBI-GeneID:''
            if any(td.get_text(strip=True) == 'NCBI-GeneID:' for td in tr.find_all(['td', 'th'])):
                # If found, get the <a> tag in the next <td> element
                next_td = tr.find('td', string=lambda x: x and 'NCBI-GeneID:' in x)
                if next_td and next_td.find_next_sibling('td'):
                    a_tag = next_td.find_next_sibling('td').find('a')
                    if a_tag and 'href' in a_tag.attrs:
                        ncbi_geneid_link = a_tag['href']
                        break
                        
        if not ncbi_geneid_link:
            return symbol or ""
        # Make the HTTP GET request to NCBI
        try:
            ncbi_response = requests.get(ncbi_geneid_link)
            ncbi_response.raise_for_status()
        except requests.HTTPError:
            return "Error fetching NCBI data."

        ncbi_soup = BeautifulSoup(ncbi_response.content, 'html.parser')
        also_known_as_dd = ncbi_soup.find('dt', string='Also known as').find_next('dd')

        synonyms = []
        if also_known_as_dd:
            also_known_as_text = also_known_as_dd.get_text(strip=True)
            synonyms = [s.strip() for s in also_known_as_text.split(';')]
        # Append symbol to synonyms if it exists
        if symbol:
            synonyms.append(symbol)
        return ', '.join(synonyms) or "No additional information found."
    # FOR IDs with SAOUHSC_*****
    elif gene_id[0] == 'S':
        response = requests.get(url_aureowiki)

        aureo_soup = BeautifulSoup(response.content, 'lxml')
        symbol = [s.get_text(strip=True)
                  for s in aureo_soup.find_all('span', class_='pan_gene_symbol')
                  if s.get_text(strip=True) != gene_id]

        if symbol:
            symbol = list(set(symbol))
            return symbol[0]
        else:
            return('')
            
def get_fulltext(pmid: str) -> str:
    """
    Retrieves the full text of an article given its PMID.

    Args:
        pmid (str): The PMID of the article.

    Returns:
        str: The full text of the article if available; an empty string otherwise.
    """
    try:
        url = metapub.FindIt(pmid).url
        urlretrieve(url, './temp')
        full_text = textract.process('./temp', extension='pdf', method='pdftotext', encoding="utf_8")
    except Exception as e:
        print(f"Error fetching full text: {e}")
        return ''
    return full_text.decode('utf-8')
    
def get_pmid_for_pmcid(pmcid: str, email: str) -> Union[str, None]:
    """
    Fetches the PMID corresponding to a given PMCID.

    Args:
        pmcid (str): The PMCID for which to find the corresponding PMID.
        email (str): Email address to use in the query (required by NCBI).

    Returns:
        str: The PMID corresponding to the given PMCID, or None if an error occurs.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0?tool=my_tool&email={email}&ids=PMC{pmcid}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        return json.loads(response.text)["records"][0]["pmid"]
    else:
        print("Error fetching the article")
        return None
        

def preprocess_texts(text:str)->str:
    """
    Preprocess a list of texts by removing punctuation, extra spaces, and converting to lowercase.

    Parameters:
    - texts (List[str]): A list of input texts.

    Returns:
    - List[str]: A list of preprocessed texts.
    """
    
    # Define punctuation to remove, preserving certain characters like '/', '_', and '-'
    punctuation = string.punctuation.replace('/', '').replace('_', '') + "∷λ"
    # Remove punctuation and extra spaces
    return text.translate(str.maketrans(punctuation, ' '*len(punctuation))).replace('  ', ' ').strip()



    



