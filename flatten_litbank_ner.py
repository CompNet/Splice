from typing import Optional, Tuple, List
import glob, os, argparse
from tqdm import tqdm
from more_itertools import flatten
from renard.pipeline.ner import NEREntity, ner_entities


def entities_are_overlapping(ent1: NEREntity, ent2: NEREntity) -> bool:
    return (ent1.start_idx >= ent2.start_idx and ent1.end_idx <= ent2.end_idx) or (
        ent2.start_idx >= ent1.start_idx and ent2.end_idx <= ent1.end_idx
    )


def filter_entity(entity: NEREntity) -> Optional[NEREntity]:
    """
    Returns the entity (or a modified version) if it should be
    accepted, or None otherwise.
    """
    if entity.tokens[0].lower() == "the":
        return filter_entity(
            NEREntity(
                entity.tokens[1:], entity.start_idx + 1, entity.end_idx, entity.tag
            )
        )

    if any([token == "," for token in entity.tokens]):
        comma_index = entity.tokens.index(",")
        return filter_entity(
            NEREntity(
                entity.tokens[:comma_index],
                entity.start_idx,
                entity.start_idx + comma_index,
                entity.tag,
            )
        )

    if any(
        [
            token[0].islower() and not token.lower() in ["the", "of"]
            for token in entity.tokens
        ]
    ):
        return None

    return entity


def extract_entities(book_path: str) -> Tuple[List[List[str]], List[List[NEREntity]]]:

    with open(book_path) as f:
        raw_text = f.read()

    # one list of the form [(token, tag1, tag2, tag3, tag4), ...] per sent
    sent_cols_lst = []
    cur_cols = []
    for line in raw_text.split("\n"):
        if line == "":
            sent_cols_lst.append(cur_cols)
            cur_cols = []
            continue
        cur_cols.append(line.split("\t"))
    if len(cur_cols) != 0:
        sent_cols_lst.append(cur_cols)

    # extract tokens and entities for each columns
    all_tokens = []
    all_entities = []

    for sent_cols in sent_cols_lst:
        tokens = [cols[0] for cols in sent_cols]
        all_tokens.append(tokens)

        tag_cols = [[line[i] for line in sent_cols] for i in range(1, 5)]
        col_entities = [ner_entities(tokens, tags) for tags in tag_cols]
        col_entities = [
            [entity for entity in entities if entity.tag == "PER"]
            for entities in col_entities
        ]

        entities_clusters = []

        # NOTE: very naive, terrible performance
        for entities in col_entities:
            other_col_entities = [lst for lst in col_entities if not lst == entities]
            other_col_entities = list(flatten(other_col_entities))

            for entity in entities:
                entity = filter_entity(entity)
                if entity is None:
                    continue

                # whether the entity was assigned to a cluster or not
                assigned = False

                # cluster assignation
                for entity_cluster in entities_clusters:
                    if any(
                        [
                            entities_are_overlapping(entity, other_entity)
                            for other_entity in entity_cluster
                        ]
                    ):
                        entity_cluster.append(entity)
                        assigned = True
                        break

                # the entity did not belong to any cluster: create its own cluster
                if not assigned:
                    entities_clusters.append([entity])

        entities = [
            max(cluster, key=lambda ent: len(ent.tokens))
            for cluster in entities_clusters
        ]
        entities = sorted(entities, key=lambda entity: entity.start_idx)

        all_entities.append(entities)

    return all_tokens, all_entities


def write_conll_(path: str, tokens: List[List[str]], entities: List[List[NEREntity]]):

    out = []

    for sent_tokens, sent_entities in zip(tokens, entities):

        sent_tags = ["O"] * len(sent_tokens)
        for entity in sent_entities:
            entity_len = entity.end_idx - entity.start_idx
            sent_tags[entity.start_idx : entity.end_idx] = ["B-PER"] + ["I-PER"] * (
                entity_len - 1
            )

        for token, tag in zip(sent_tokens, sent_tags):
            out.append(f"{token} {tag}")

        out.append("")

    with open(path, "w") as f:
        f.write("\n".join(out))


def flatten_litbank_ner_(litbank_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for book in tqdm(glob.glob(f"{litbank_dir}/entities/tsv/*.tsv")):
        tokens, entities = extract_entities(book)
        conll_name = os.path.splitext(os.path.basename(book))[0] + ".conll"
        write_conll_(f"{output_dir}/{conll_name}", tokens, entities)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, help="path to the litbank repository."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="path the the output directory. It will be created if needed.",
    )
    args = parser.parse_args()

    flatten_litbank_ner_(args.input_dir, args.output_dir)
