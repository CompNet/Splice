from __future__ import annotations
from typing import List
import pathlib as pl
from dataclasses import dataclass
from renard.ner_utils import load_conll2002_bio
from renard.gender import Gender
from renard.pipeline.ner import NEREntity
from renard.pipeline.character_unification import Character
from tibert.bertcoref import Mention, CoreferenceDataset


@dataclass
class Novel:
    title: str
    tokens: List[str]
    sents: List[List[str]]
    entities: List[NEREntity]
    corefs: List[List[Mention]]
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


def load_novel(
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
