from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
import uuid
import pandas as pd

class RDFCreator:
    """
    A class for creating RDF (Resource Description Framework) graphs from structured data contained within a DataFrame.
    It establishes RDF triples to represent relationships between entities (such as sRNAs and target genes) and their attributes,
    facilitating semantic web or linked data applications.

    The class uses predefined RDFLib Namespaces for consistent identification of entities and relationships within the RDF graph.
    """
    def __init__(self):
        """
        Initializes the RDFCreator, setting up namespaces and preparing an RDF graph for triple storage.

        Args:
            df (pd.DataFrame): The DataFrame containing regulation statements and entity information.

        Attributes:
            df (pd.DataFrame): Stores the input DataFrame.
            WD, P, EX, PMC (Namespace): RDFLib Namespaces for Wikidata entities, properties, example resources, and PMC articles.
            g (Graph): An RDFLib Graph object for storing RDF triples.
        """
        self.WD = Namespace("http://www.wikidata.org/entity/")
        self.P = Namespace("http://www.wikidata.org/prop/direct/")
        self.EX = Namespace("http://example.org/resource/")
        self.PMC = Namespace("https://www.ncbi.nlm.nih.gov/pmc/articles/")
        self.g = Graph()
        self.g.bind("wd", self.WD)
        self.g.bind("p", self.P)
        self.g.bind("ex", self.EX)
        self.g.bind("pmc", self.PMC)

    def create_rdf(self, df:pd.DataFrame):
        """
        Converts data from a DataFrame into RDF triples and adds them to the RDF graph. This method specifically
        focuses on representing regulation statements, including entities like sRNAs and target genes, their
        relationships, and associated metadata like PMCIDs and textual descriptions.

        Args:
            df (pd.DataFrame): A DataFrame containing columns for 'sRNA_QIDs', 'TargetGene_QIDs', 'relation_prop',
                               possibly 'pmcid', and 'text_prep' to represent regulation statements and their context.
        """
        for index, row in df.iterrows():
            # Generate a unique identifier for each statement to ensure URI uniqueness
            statement_uri = URIRef(f"{self.EX}{str(uuid.uuid4())[:8]}")

            # Create URIs for sRNA, target gene, and the regulation relation type
            srna_uri = URIRef(self.WD[row['sRNA_QIDs']])
            targetgene_uri = URIRef(self.WD[row['TargetGene_QIDs']])

            # Add triples for the regulation relationship, regulated entity, regulator, and target gene
            self.g.add((statement_uri, RDF.type, self.EX.RegulationStatement))
            self.g.add((statement_uri, self.P[row["relation_prop"]], targetgene_uri))
            self.g.add((statement_uri, self.EX.hasRegulator, srna_uri))
            self.g.add((statement_uri, self.EX.hasTargetGene, targetgene_uri))

            # Optionally add the source PMCID and descriptive text as triples, if available
            if "pmcid" in row and row["pmcid"]:
                pmcid_uri = URIRef(self.PMC[row["pmcid"]])
                self.g.add((statement_uri, self.EX.hasSource, pmcid_uri))
            if "text_prep" in row and row["text_prep"]:
                self.g.add((statement_uri, RDFS.comment, Literal(row["text_prep"])))

        return self.g
