# Extraction of Bacterial sRNA Regulation Statements

## Introduction
This project explores the application of Named Entity Recognition (NER) techniques to extract small RNA (sRNA) entities and their interaction partners from biomedical literature, aiming to advance understanding of sRNA-mediated gene regulation. Leveraging sophisticated tools like SciSpaCy and BioBERT, an ensemble model was developed to achieve high accuracy in entity recognition. Additionally, this study addresses the classification of relationships between sRNA entities and their interaction partners, mapping recognized entities to Wikidata for validation, and structuring extracted statements in RDF format for integration into knowledge graphs.

## Installation
Instructions on setting up the project environment can be found in the `requirements.txt` file, which lists all the necessary Python libraries and dependencies. Ensure you have Python installed on your system and then install the requirements using pip:
```bash
pip install -r requirements.txt
```

### Scripts and Notebooks
- `Data_Acquisition_sRNA_TarBase.ipynb`: Fetches and preprocesses sRNA interaction data.
- `data_preparation.ipynb`: Prepares data for analysis, including cleaning and formatting.
- `extract_regulation_statements.py`: Extracts regulation statements from the processed data.
- `FineTune_BioBert_NER.ipynb`: Fine-tunes BioBERT for NER tasks specific to sRNA entities.
- `FineTune_Spacy_NER.ipynb`: Demonstrates the fine-tuning of SpaCy models for sRNA NER.
- `relation_classification_biobert.ipynb`: Fine-tunes BioBERT for classifying relationships between entities.
- `relation_classification_using_NER_Tags.ipynb`: Fine-tunes BioBERT enhanced by NER tags to aid in relationship classification.
- `relation_extraction.ipynb`: Details the methodology for extracting relational data from texts.
- `SciSpacy_BioBert_Ensemble.ipynb`: Showcases the development and performance of the ensemble NER model.
- `training_utils.py`, `preprocessing_utils.py`, `negative_examples.py`, `WDReferenceFetcher.py`: Utility scripts supporting model training, data preprocessing, and entity validation.

## Features
- High-accuracy NER leveraging an ensemble of SciSpaCy and BioBERT models.
- Efficient relationship classification strategy reducing computational resources.
- Integration of recognized entities with Wikidata for validation.
- RDF statement structuring for knowledge graph integration.

## Usage
This project consists of multiple components that together facilitate the extraction and classification of sRNA entities and their relationships. The final script that ties the analyses together is extract_regulation_statements.py.
The paths within the notebooks and scripts need to be adjusted before running.

### Running extract_regulation_statements.py
To run this script, you need to specify the input format, input path, and output path at a minimum. 
If processing a single document in txt format containing the fulltext, you also need to provide the PubMed Central ID (PMCID) for the source document.
If processing in bulk mode, the input format is a `parquet` file of a pandas DataFrame including at least the two columns `"PMCID"` and `"fulltext"`

```bash
python extract_regulation_statements.py --input_format [single|bulk] --input_path "<path_to_input>" --pmcid "<PMCID>" --output_path "<path_to_output>"
```
Replace `[single|bulk]` with either single for single file processing or bulk for bulk processing. `<path_to_input>` and `<path_to_output>` should be replaced with the actual paths to your input and output files. If the input format is single, replace `<PMCID>` with the actual PubMed Central ID..

## Dependencies
All requirements are listed in the requirements.txt file. Ensure all dependencies are installed to avoid runtime errors.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
