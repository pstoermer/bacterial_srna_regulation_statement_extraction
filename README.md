# Extraction of Bacterial sRNA Regulation Statements

## Introduction
This project explores the application of Named Entity Recognition (NER) techniques to extract small RNA (sRNA) entities and their interaction partners from biomedical literature, aiming to advance understanding of sRNA-mediated gene regulation. Leveraging sophisticated tools like SciSpaCy and BioBERT, an ensemble model was developed to achieve high accuracy in entity recognition. Additionally, this study addresses the classification of relationships between sRNA entities and their interaction partners, mapping recognized entities to Wikidata for validation, and structuring extracted statements in RDF format for future integration into knowledge graphs.

## Installation
Instructions on setting up the project environment can be found in the `requirements.txt` file, which lists all the necessary Python libraries and dependencies. Ensure you have Python installed on your system and then install the requirements using pip:
```bash
pip install -r requirements.txt
```

### Scripts and Notebooks

## Repository Structure
- `01_src`: Contains notebooks and scripts for data acquisition, preprocessing, model training, and other analyses related to the study.
  - `Data_Acquisition_sRNA_TarBase.ipynb`: Fetches and preprocesses sRNA interaction data.
  - `data_preparation.ipynb`: Prepares data for analysis, including cleaning and formatting.
  - `extract_regulation_statements.py`: Extracts regulation statements from the processed data.
  - `FineTune_BioBert_NER.ipynb`: Fine-tunes BioBERT for NER tasks specific to sRNA entities.
  - `FineTune_Spacy_NER.ipynb`: Demonstrates the fine-tuning of SpaCy models for sRNA NER.
  - `relation_classification_biobert.ipynb`: Fine-tunes BioBERT for classifying relationships between entities.
  - `relation_classification_using_NER_Tags.ipynb`: Fine-tunes BioBERT enhanced by NER tags to aid in relationship classification.
  - `relation_extraction.ipynb`: Details the methodology for extracting relational data from texts.
  - `SciSpacy_BioBert_Ensemble.ipynb`: Showcases the development and performance of the ensemble NER model.
  -  `WDReferenceFetcher.py`: Fetches Wikidata references for InteractOA data.
  - `training_utils.py`, `preprocessing_utils.py`, `negative_examples.py`: Utility scripts supporting model training, data preprocessing, and entity validation.
- `02_queries`: Includes queries used for data acquisition.
    - Query files for WDReferenceFetcher.py
- `03_srna_statement_extractor`: Houses the modularized sRNA Statement Extractor, encompassing all necessary classes for the extraction pipeline, alongside the configuration file.



## Features
- Ensemble NER model combining SciSpaCy and BioBERT for accurate entity recognition.
- Relationship classification with minimal computational overhead.
- Entity validation and mapping against Wikidata.
- Structuring extracted statements in RDF for knowledge graph enrichment.
## Usage
### sRNA Statement Extractor
The `03_srna_statement_extractor` directory features the advanced sRNA statement extraction module, built upon the initial extract_regulation_statements.py script. This module incorporates sophisticated NER, relationship classification, and RDF serialization techniques, all configurable through a YAML file.

Configuration
Before execution, ensure a `config.yml` file is present within the `03_srna_statement_extractor` directory or specified directory, detailing model paths.

Execution
Run the script with the following command, providing appropriate arguments for your data and configuration:
```bash
python extract_regulation_statements.py \
  --config_path "<path_to_config>" \
  --input_format [single|bulk] \
  --input_path "<path_to_input>" \
  --pmcid "<PMCID_if_applicable>" \
  --output_path "<path_to_output>"
```

- `--config_path`: Path to the YAML configuration file.
- `--input_format`: 'single' for processing a single text file or 'bulk' for a DataFrame in a parquet file.
- `--input_path`: File path for the input document(s).
- `--pmcid`: Required if the input format is "single"; the PubMed Central ID of the source document.
- `--output_path`: Destination path for the serialized RDF output.

## Dependencies
All requirements are listed in the requirements.txt file. Ensure all dependencies are installed to avoid runtime errors.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
