#!/usr/bin/env python
"""
Nanopub Batch Upload GitHub Action Script

TODO:
- add ability to restrict model to NamedThing if graph is too large
- figure out when default namespace is used and when model id
"""

import sys
import click
import itertools
import logging
import nanopub
import nanopub.definitions
import pathlib
import rdflib
import re
import requests
import yaml

from pathlib import Path
from rdflib.namespace import SKOS, RDF
from typing import List, Optional, Mapping

from linkml_runtime.utils.schemaview import SchemaView
from linkml_runtime.loaders import YAMLLoader
from linkml_runtime.dumpers import YAMLDumper
from linkml_runtime.dumpers import RDFLibDumper
from linkml.generators.pythongen import PythonGenerator
from linkml_runtime.utils.yamlutils import YAMLRoot

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


BASE_NAMESPACE = rdflib.Namespace("https://w3id.org/peh/terms/")


class NanopubGenerator:
    def __init__(
        self,
        orcid_id: str,
        name: str,
        private_key: str,
        public_key: str,
        intro_nanopub_uri: str,
        test_server: bool,
    ):
        self.profile = nanopub.Profile(
            orcid_id=orcid_id,
            name=name,
            private_key=private_key,
            public_key=public_key,
            introduction_nanopub_uri=intro_nanopub_uri,
        )

        self.np_conf = nanopub.NanopubConf(
            profile=self.profile,
            use_test_server=test_server,
            add_prov_generated_time=True,
            attribute_publication_to_profile=True,
        )

    def create_nanopub(self, assertion: rdflib.Graph) -> nanopub.Nanopub:
        return nanopub.Nanopub(conf=self.np_conf, assertion=assertion)

    def update_nanopub(self, np_uri: str, assertion: rdflib.Graph) -> nanopub.Nanopub:
        new_np = nanopub.NanopubUpdate(
            uri=np_uri,
            conf=self.np_conf,
            assertion=assertion,
        )
        new_np.sign()
        return new_np

    @classmethod
    def is_nanopub_id(cls, key: str):
        allowed_prefixes = [
            "http://purl.org",
            "https://purl.org",
            "http://w3id.org",
            "https://w3id.org",
        ]
        for prefix in allowed_prefixes:
            if key.startswith(prefix):
                return True
        return False

    def check_nanopub_existence(self, entity: YAMLRoot) -> bool:
        try:
            # np_conf = self.np_conf
            url = getattr(entity, "id", None)
            if url is not None:
                return self.is_nanopub_id(url)
            else:
                raise ValueError("Entity id is None.")

        except Exception as e:
            logger.error(f"Error in check_nanopub_existence: {e}")

    def publish_single(
        self,
        to_publish: rdflib.Graph,
        supersedes: Optional[str] = None,
        dry_run: bool = True,
    ) -> str:
        try:
            if supersedes is None:
                np = self.create_nanopub(assertion=to_publish)
                np.sign()
                np_uri = np.metadata.np_uri
                if np_uri is None:
                    raise ValueError("no URI returned by nanpub server.")
                if not dry_run:
                    publication_info = np.publish()
                    logger.info(f"Nanopub published: {publication_info}")
            else:
                raise NotImplementedError

            return np_uri

        except Exception as e:
            logger.error(f"Error in publish_single: {e}")
            raise

    def publish_sequence(
        self,
        to_publish: list,
        supersedes: Optional[list[str]] = None,
        dry_run: bool = True,
    ) -> list:
        try:
            np_uris = []
            if supersedes is None:
                supersedes = []
            for statement, supersedes_uri in itertools.zip_longest(
                to_publish, supersedes
            ):
                np_uri = self.publish_single(statement, dry_run=dry_run)
                np_uris.append(np_uri)

            return np_uris

        except Exception as e:
            logger.error(f"Error in publish_sequence: {e}")
            raise


def load_yaml(
    schema_path: str,
    data_path: str,
) -> YAMLRoot:
    try:
        # Load schema
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        schema_view = SchemaView(str(schema_path))

        # Generate Python classes from schema
        python_module = PythonGenerator(str(schema_path)).compile_module()
        target_class = "EntityList"  # This could be made configurable

        if target_class not in python_module.__dict__:
            raise ValueError(f"Target class '{target_class}' not found in schema")

        py_target_class = python_module.__dict__[target_class]

        # Load data
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # data is instance of EntityList dataclass
        data = YAMLLoader().load(str(data_path), py_target_class)

        return data, schema_view

    except Exception as e:
        logger.error(f"Error in load_yaml: {e}")
        raise


def process_yaml_root(
    root: YAMLRoot,
    target_name: str,
) -> List:
    try:
        # Process entities
        target_data_list = getattr(root, target_name, None)
        if target_data_list is None:
            raise ValueError(f"Target list '{target_name}' not found in data dict")

        return target_data_list

    except Exception as e:
        logger.error(f"Error in process_yaml_root: {e}")
        raise


def get_property_mapping(
    data: List, schema_view: SchemaView, base: rdflib.Namespace
) -> Mapping:
    """
    Mapping of the kind: {property_name: slot_uri}
    example: {'name': rdflib.term.URIRef('http://www.w3.org/2004/02/skos/core#altLabel')}
    """
    namespace_mapping = {}
    for entity in data:
        if getattr(entity, "translations") is not None:
            for translation in entity.translations:
                if translation.property_name not in namespace_mapping:
                    property_name = translation.property_name
                    slot_def = schema_view.all_slots().get(property_name)
                    curie_str = getattr(slot_def, "slot_uri")
                    if curie_str is None:
                        curie_str = base[property_name]
                    uri_str = schema_view.expand_curie(curie_str)
                    namespace_mapping[property_name] = rdflib.term.URIRef(uri_str)

    return namespace_mapping


def add_translation_to_graph(
    g: rdflib.Graph, property_mapping: Mapping
) -> rdflib.Graph:
    try:
        if len(property_mapping) == 0:
            logger.info("LinkML schema does not contain translations.")
            return g

        #  Iterate over the triples and perform the transformation and removal
        for s, _, o in g.triples((None, BASE_NAMESPACE.translations, None)):
            language = g.value(o, BASE_NAMESPACE.language)
            property_name = str(g.value(o, BASE_NAMESPACE.property_name))
            translated_value = g.value(o, BASE_NAMESPACE.translated_value)
            # Apply the mapping
            if property_name in property_mapping:
                mapped_property = property_mapping[property_name]
                g.add(
                    (
                        s,
                        mapped_property,
                        rdflib.Literal(translated_value, lang=language),
                    )
                )

            # Remove the unnecessary blank node triples
            g.remove((o, None, None))
            g.remove((None, None, o))

        return g

    except Exception as e:
        logging.error(f"Error in add_translation_to_graph: {e}")
        raise


def add_vocabulary_membership(
    g: rdflib.Graph, vocab_uri: str, subject_type: rdflib.URIRef
) -> rdflib.Graph:
    """
    Adds vocabulary membership information to each concept in the graph.

    Args:
        g: An rdflib Graph instance containing vocabulary terms
        vocab_uri: URI string of the vocabulary collection

    Returns:
        The modified graph with vocabulary membership added
    """
    try:
        # Create a URI reference for the vocabulary
        vocabulary = rdflib.URIRef(vocab_uri)
        concepts = list(g.subjects(RDF.type, subject_type))
        SKOS_COLLECTION = SKOS.inScheme
        # Add the membership triple to each concept
        for concept in concepts:
            g.add((concept, SKOS_COLLECTION, vocabulary))

        return g
    except Exception as e:
        logging.error(f"Error in add_vocabulary_membership: {e}")
        raise


def yaml_dump(root: YAMLRoot, target_name: str, entities: List, file_name: str):
    # Use setattr to update the target field
    setattr(root, target_name, entities)
    return YAMLDumper().dump(root, to_file=file_name)


def extract_id(url: str, type_prefix: Optional[str] = None):
    """Extract the type prefix (MA, UN, etc.) and the ID from a w3id.org URL."""
    match = re.search(rf"w3id\.org/peh/{type_prefix}-([a-f0-9]+)", url)
    if match:
        return match.group(1)
    return None


def generate_htaccess(redirects: List, type_prefix: str):
    """Generate .htaccess content."""

    rules = []

    for source, target in redirects:
        local_path = extract_id(source, type_prefix)
        if local_path is None:
            logger.error(
                "Error in generate_htaccess: no id could be extracted from uri."
            )
            sys.exit(1)
        if local_path:
            rules.append(f"RewriteRule ^{local_path}$ {target} [R=302,L]")

    return "\n".join(rules)


def update_htaccess(
    redirects: List, output_file: str, type_prefix: Optional[str] = None
):
    # example header
    # """Generate or update an .htaccess file."""
    # header = """RewriteEngine On
    #
    ## PEH redirections
    ## Format: Local ID to nanopub
    # """

    if not redirects:
        print("No valid redirects found in input file.", file=sys.stderr)
        sys.exit(1)

    new_content = generate_htaccess(redirects, type_prefix=type_prefix)
    with open(output_file, "w") as f:
        f.write(new_content)

    print(f"Successfully wrote .htaccess to {output_file}")
    print(f"Added {len(redirects)} redirect rules")


def dump_identifier_pairs(pairs: List[tuple], file_name: str):
    try:
        with open(file_name, "w") as outfile:
            for pair in pairs:
                w3id_uri, nanopub_uri = pair
                print(f"{w3id_uri}, {nanopub_uri}", file=outfile)
    except Exception as e:
        logging.error(f"Error in dump_identifier_pairs: {e}")
        raise


def is_valid_assertion_graph(g: rdflib.Graph) -> bool:
    # TODO: add more checks
    return len(g) > 0


def build_rdf_graph(
    entity: "YAMLRoot",
    schema_view: SchemaView,
    translation_namespace_mapping: Optional[Mapping] = None,
    vocab_uri: Optional[str] = None,
) -> rdflib.Graph:
    """
    Convert a LinkML entity to an RDF graph.

    Args:
        entity: The LinkML entity to convert
        schema_view: The schema view defining the entity structure

    Returns:
        An RDF graph representing the entity
    """
    try:
        rdf_string = RDFLibDumper().dumps(entity, schema_view)
        g = rdflib.Graph()
        g.parse(data=rdf_string)
        assert len(g) < nanopub.definitions.MAX_TRIPLES_PER_NANOPUB
        # ADD vocabulary membership
        if vocab_uri is not None:
            entity_class_name = entity.__class__.__name__
            # example: subject_type = SKOS.Concept
            class_curie = schema_view.get_uri(entity_class_name)
            class_uri = schema_view.expand_curie(class_curie)
            subject_type = rdflib.term.URIRef(class_uri)
            g = add_vocabulary_membership(
                g, vocab_uri=vocab_uri, subject_type=subject_type
            )
        # ADD TRANSLATION
        if translation_namespace_mapping is not None:
            g = add_translation_to_graph(g, translation_namespace_mapping)
        if is_valid_assertion_graph(g):
            return g
        else:
            raise AssertionError("Assertion Graph is invalid.")
    except Exception as _:
        logger.error("Error converting entity to RDF:", exc_info=True)
        logger.debug("Entity details: %s", entity)
        logger.debug(
            "Additional context: vocab_uri=%s, translation_namespace_mapping=%s",
            vocab_uri,
            translation_namespace_mapping,
        )
        raise


@click.command()
@click.option(
    "--schema",
    "-s",
    "schema_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the LinkML schema file",
)
@click.option(
    "--data",
    "-d",
    "data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML data file",
)
@click.option(
    "--target",
    "-t",
    "target_name",
    required=True,
    help="Name of the target entity list in the data file",
)
@click.option(
    "--changelog",
    "changelog_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the changelog file.",
)
@click.option(
    "--orcid-id",
    required=True,
    envvar="NANOPUB_ORCID_ID",
    help="ORCID ID for nanopub profile",
)
@click.option(
    "--name", required=True, envvar="NANOPUB_NAME", help="Name for nanopub profile"
)
@click.option(
    "--private-key",
    required=True,
    envvar="NANOPUB_PRIVATE_KEY",
    help="Private key for nanopub profile",
)
@click.option(
    "--public-key",
    required=True,
    envvar="NANOPUB_PUBLIC_KEY",
    help="Public key for nanopub profile",
)
@click.option(
    "--intro-nanopub-uri",
    required=True,
    envvar="NANOPUB_INTRO_URI",
    help="Introduction nanopub URI",
)
@click.option("--dry-run", is_flag=True, help="Prepare nanopubs but do not publish")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--output-pairs",
    "output_path_pairs",
    required=False,
    type=click.Path(),
    help="Path to output identifier nanopub pairs",
    default=None,
)
@click.option(
    "--vocab",
    "vocab_uri",
    required=False,
    type=str,
    help="URI for the larger vocabulary this term is part of.",
    default=None,
)
@click.option(
    "--type-prefix",
    "type_prefix",
    required=False,
    type=str,
    default=None,
    help="Vocabulary-specific prefix for uri to be generated.",
)
@click.option(
    "--preflabel",
    "preflabel",
    required=False,
    default="label",
    type=str,
    help="Key to human readable identifier field for resource.",
)
def main(
    schema_path: str,
    data_path: str,
    target_name: str,
    changelog_path: str,
    orcid_id: str,
    name: str,
    private_key: str,
    public_key: str,
    intro_nanopub_uri: str,
    dry_run: bool = True,
    verbose: bool = False,
    output_path_pairs: str = None,
    vocab_uri: str = None,
    type_prefix: str = None,
    id_key: str = "id",
    preflabel: str = "label",
):
    """
    Create and publish nanopublications from structured data.

    This tool takes data structured according to a LinkML schema and publishes
    it as nanopublications. It's designed to be run as part of a GitHub Actions
    workflow, with authentication details provided as GitHub secrets.
    """
    # Set logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)

    try:
        identifier_pairs = []
        # Count for reporting
        processed = 0
        published = 0
        updated = 0
        label_action_map = {}
        implemented_actions = set(["added", "modified"])

        assert dry_run

        # load data formatted according to peh linkml schema
        logger.info(f"Processing data from {data_path} using schema {schema_path}")
        yaml_root, schema_view = load_yaml(schema_path, data_path)
        entities = process_yaml_root(yaml_root, target_name)

        click.echo("Loading changelog ...")
        with open(changelog_path, "r") as f:
            changelog = yaml.safe_load(f)

        for change in changelog["changes"]:
            action = change["action"]
            label = change.get(preflabel)
            label_action_map[label] = action
            if action not in implemented_actions:
                logger.error(f"Action {change} currently not implemented")
                raise NotImplementedError

        ## REQUIREMENTS PRIOR TO PUBLISHING
        # make namespace mapping for language annotation purposes
        namespace_mapping = get_property_mapping(entities, schema_view, BASE_NAMESPACE)
        if len(namespace_mapping) == 0:
            namespace_mapping = None

        nanopub_generator = NanopubGenerator(
            orcid_id=orcid_id,
            name=name,
            private_key=private_key,
            public_key=public_key,
            intro_nanopub_uri=intro_nanopub_uri,
            test_server=dry_run,
        )

        ## START PUBLISHING
        for entity in entities:
            label = getattr(entity, preflabel)
            if label in label_action_map:
                action = label_action_map[label]
                entity_id = getattr(entity, id_key)
                if action == "added":
                    ## Entity has no nanopub yet
                    peh_uri = entity_id
                    graph = build_rdf_graph(
                        entity, schema_view, namespace_mapping, vocab_uri=vocab_uri
                    )
                    processed += 1
                    np_uri = nanopub_generator.publish_single(graph, dry_run=dry_run)
                    published += 1
                    # create w3id - nanopub pairs
                    identifier_pairs.append((peh_uri, str(np_uri)))
                    logger.info(f"Term {peh_uri}: nanopub {np_uri}")

                elif action == "modified":
                    response = requests.get(entity_id, allow_redirects=True, timeout=10)
                    if not response.status_code == 200:
                        logger.error(
                            f"Voc entry {entity_id} could not be redirected for update."
                        )
                        continue
                    current_np_uri = response.url
                    np_uri = nanopub_generator.publish_single(
                        graph, supersedes=current_np_uri, dry_run=dry_run
                    )
                    updated += 1
                    # create w3id - nanopub pairs
                    identifier_pairs.append((peh_uri, str(np_uri)))
                    logger.info(f"Term {peh_uri}: nanopub {np_uri}")

                else:
                    raise NotImplementedError

        # Report summary
        logger.info(
            f"Processing complete. Processed: {processed}, "
            f"Published: {published}, Updated: {updated}"
        )

        # dump identifier_pairs
        if output_path_pairs is None:
            output_path_pairs = "./pairs.txt"
        output_path_pairs = pathlib.Path(output_path_pairs).resolve()
        _ = update_htaccess(identifier_pairs, output_path_pairs, type_prefix)

    except Exception as e:
        logger.error(f"Error in processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
