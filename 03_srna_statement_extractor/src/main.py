import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

from config_manager import ConfigManager
from data_loader import DataLoader
from text_processor import TextProcessor
from ner_extractor import NERExtractor
from entity_processor import EntityProcessor
from relation_classifier import RelationClassifier
from wikidata_mapper import WikidataMapper
from rdf_creator import RDFCreator


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the processing pipeline.

    Returns:
        argparse.Namespace: The parsed arguments including the configuration file path,
        input format, input path, PMC ID (if applicable), and output path.
    """

    parser = argparse.ArgumentParser(description='Extract sRNA regulation statements.')
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help='Path to the configuration file.'
    )

    parser.add_argument(
        "--input_format",
        type=str,
        required=True,
        choices=['single', 'bulk'],
        help='Type of input to process (single txt file or bulk process a parquet file containing a pd.DataFrame'
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input file."
    )

    parser.add_argument(
        "--pmcid",
        type=str,
        help='PubMed Central ID for the source document (required if input format is "single").'
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the RDF output.")

    return parser.parse_args()


def main():
    """
    Main function orchestrating the regulation statement extraction process.

    This function integrates the configuration management, data loading, text preprocessing,
    relation classification, named entity recognition, entity processing, Wikidata mapping,
    and RDF creation into a cohesive pipeline.
    """
    # Parse input arguments
    args = parse_arguments()

    # Initialize configuration manager
    config_path = Path(args.config_path)
    config_manager = ConfigManager(config_path)

    # Load input data
    data_loader = DataLoader(input_format=args.input_format, input_path=args.input_path, pmcid=args.pmcid)
    df = data_loader.load_input()

    # Preprocess texts
    preprocessor = TextProcessor()
    df = preprocessor.preprocess_texts(df)

    df = df.iloc[:2000]
    # Classify relations
    relation_classifier = RelationClassifier(model_path=config_manager.get('relation_classifier'))
    df['relation_name'] = df.text_prep.apply(relation_classifier.classify)

    # Remove rows that were classified with relation type 'none'
    df = df[df.relation_name != 'none']
    print(df)
    # Extract named entities
    ner_extractor = NERExtractor(
        biobert_model_path=config_manager.get('biobert'),
        scispacy_model_path=config_manager.get('scispacy'),
        id_to_label_path=config_manager.get('id_to_label'))
    df['combined_entities'] = df.text_prep.progress_apply(ner_extractor.extract_entities)

    df = df[df['combined_entities'].apply(lambda x: len(x) > 0)]
    df = df.reset_index()
    # Process entities
    entity_processor = EntityProcessor()
    df = entity_processor.separate_entities(df, combined_entities_col="combined_entities")
    df = entity_processor.filter_empty_entities(df)

    df['sRNA Entities'] = df['sRNA Entities'].apply(entity_processor.clean_entities)
    df['TargetGene Entities'] = df['TargetGene Entities'].apply(entity_processor.clean_entities)

    df = entity_processor.filter_cross_listed_entities(df)
    df = entity_processor.filter_empty_entities(df)
    print(df)
    # Query Wikidata for entity QIDs
    wikidata_mapper = WikidataMapper()
    unique_srnas, unique_genes = wikidata_mapper.generate_unique_entity_lists(df)
    srna_qid_map = wikidata_mapper.query_entities_for_qids(unique_srnas, 'srna')
    gene_qid_map = wikidata_mapper.query_entities_for_qids(unique_genes, 'gene')

    # Map entities to QIDs and filter
    df = wikidata_mapper.map_entity_qids_and_filter(df, srna_qid_map, gene_qid_map)
    df['relation_prop'] = df['relation_name'].apply(wikidata_mapper.map_relation_to_wikidata)
    df = df.apply(wikidata_mapper._refine_entities, axis=1).dropna().reset_index(drop=True)
    df = entity_processor.filter_empty_entities(df)
    df = wikidata_mapper.expand_rows(df)

    print(df)
    if not df.empty:
        # Create RDF Output
        rdf_creator = RDFCreator()
        rdf_graph = rdf_creator.create_rdf(df)

        rdf_graph.serialize(destination=args.output_path, format='xml')

    else:
        print('No regulation statements found.')


if __name__ == "__main__":
    main()




