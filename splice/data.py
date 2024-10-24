from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import os, re
import pathlib as pl
from dataclasses import dataclass
from more_itertools import flatten
from collections import defaultdict
import pandas as pd
from renard.ner_utils import load_conll2002_bio
from renard.gender import Gender
from renard.pipeline.ner import NEREntity
from renard.ner_utils import ner_entities
from renard.pipeline.character_unification import Character
from tibert.bertcoref import Mention, CoreferenceDataset


@dataclass
class Novel:
    title: str
    tokens: List[str]
    sents: List[List[str]]
    entities: List[NEREntity]
    corefs: Optional[List[List[Mention]]]
    characters: List[Character]

    def cut(self, sents_len: int) -> Novel:
        assert sents_len >= 0

        sents = self.sents[:sents_len]
        tokens = sents[0]

        entities = [ent for ent in self.entities if ent.end_idx <= len(tokens)]

        corefs = []
        for chain in self.corefs:
            chain = [m for m in chain if m.end_idx <= len(tokens)]
            if len(chain) > 0:
                corefs.append(chain)

        characters = []
        for character in self.characters:
            mentions = [m for m in character.mentions if m.end_idx <= len(tokens)]
            characters.append(Character(character.names, mentions))

        return Novel(self.title, tokens, sents, entities, corefs, characters)


def extract_characters(
    entities: List[NEREntity],
    corefs: List[List[Mention]],
    keep_only_NER_mentions: bool = False,
) -> List[Character]:
    """Extract gold characters using gold labels for NER and
    coreference

    :param only_NER_mentions: if ``True``, will only consider mentions
        resolved as PER as character mentions instead of including all
        coreferent mentions.
    """
    # we consider that a full chain form a character if at least one
    # of its constituent is an entity
    characters = []

    for chain in corefs:
        entity_mentions = [
            mention
            for mention in chain
            if any(
                mention.start_idx == entity.start_idx
                and mention.end_idx == entity.end_idx
                for entity in entities
            )
        ]

        # no PER entity in the whole chain: ignore
        if len(entity_mentions) == 0:
            continue

        names = frozenset({" ".join(mention.tokens) for mention in entity_mentions})
        if keep_only_NER_mentions:
            characters.append(Character(names, entity_mentions, Gender.UNKNOWN))
        else:
            characters.append(Character(names, chain, Gender.UNKNOWN))

    return characters


def load_litbank_novel(
    ner_path: pl.Path, coref_path: pl.Path, keep_only_NER_mentions: bool = False
) -> Novel:
    """
    :param only_NER_mentions: passed to :func:`.extract_characters`
    """
    title = ner_path.name.split(".")[0].rstrip("_brat")

    sents, tokens, entities = load_conll2002_bio(str(ner_path), separator=" ")

    coref_dataset = CoreferenceDataset.from_conll2012_file(
        str(coref_path),
        None,  # OK
        max_span_size=999,  # OK?
        tokens_split_idx=3,
        corefs_split_idx=12,
    )
    assert coref_dataset.documents[0].tokens == tokens
    coref_chains = coref_dataset.documents[0].coref_chains

    characters = extract_characters(
        entities, coref_chains, keep_only_NER_mentions=keep_only_NER_mentions
    )

    return Novel(title, tokens, sents, entities, coref_chains, characters)


def load_novelties_conll2002_bio(
    path: str,
    tag_conversion_map: Optional[Dict[str, str]] = None,
    separator: str = "\t",
    **kwargs,
) -> Tuple[List[List[str]], List[str], List[NEREntity]]:
    """Load a file under CoNLL2022 BIO format.  Sentences are expected
    to be separated by end of lines.  Tags should be in the CoNLL-2002
    format (such as 'B-PER I-PER') - If this is not the case, see the
    ``tag_conversion_map`` argument.

    :param path: path to the CoNLL-2002 formatted file
    :param separator: separator between token and BIO tags
    :param tag_conversion_map: conversion map for tags found in the
        input file.  Example : ``{'B': 'B-PER', 'I': 'I-PER'}``
    :param kwargs: additional kwargs for ``open`` (such as
        ``encoding`` or ``newline``).

    :return: ``(sentences, tokens, entities)``
    """
    tag_conversion_map = tag_conversion_map or {}

    with open(os.path.expanduser(path), **kwargs) as f:
        raw_data = f.read()

    sents = []
    sent_tokens = []
    tags = []
    for line in raw_data.split("\n"):
        line = line.strip("\n")
        try:
            token, tag = line.split(separator)
        except ValueError:
            continue
        sent_tokens.append(token)
        tags.append(tag_conversion_map.get(tag, tag))
        if token in [".", "!", "?"]:
            if len(sent_tokens) == 0:
                continue
            sents.append(sent_tokens)
            sent_tokens = []
    if not len(sent_tokens) == 0:
        sents.append(sent_tokens)

    tokens = list(flatten(sents))
    entities = ner_entities(tokens, tags)

    return sents, tokens, entities


def load_novelties_novel(
    ner_paths: List[pl.Path], alias_resolution_path: pl.Path
) -> Novel:
    title = ner_paths[0].name.split(".")[0]

    # extract NER data
    novel_sents = []
    novel_tokens = []
    novel_entities = []
    for ner_path in ner_paths:
        sents, tokens, entities = load_novelties_conll2002_bio(
            str(ner_path), separator=" "
        )
        novel_sents += sents
        novel_tokens += tokens
        novel_entities += entities

    # entities are noted CHR in novelties, but Splice uses PER => we
    # fix entity tags
    for ent in novel_entities:
        if ent.tag == "CHR":
            ent.tag = "PER"

    # extract alias resolution data
    # { canonical form => form }
    aliases = defaultdict(set)
    ardf = pd.read_csv(alias_resolution_path)
    for i, row in ardf.iterrows():
        if i == 0:
            title = row["Metadata"][6:]
        aliases[row["Form"]].add(row["Entity"])

    # parse alias resolution data into characters
    characters = []
    for _, forms in aliases.items():
        mentions = []
        for entity in novel_entities:
            if entity.tag != "PER":
                continue
            if " ".join(entity.tokens) in forms:
                mentions.append(
                    Mention(entity.tokens, entity.start_idx, entity.end_idx)
                )
        characters.append(Character(frozenset(forms), mentions))

    return Novel(title, novel_tokens, novel_sents, novel_entities, None, characters)
