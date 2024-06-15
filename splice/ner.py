from __future__ import annotations
import random
from copy import deepcopy
from typing import Optional, Callable
from renard.pipeline.ner import NEREntity
from tibert.utils import spans_indexs


def entities_to_BIO(tokens: list[str], entities: list[NEREntity]) -> list[str]:
    """Convert a list of entities to BIO tags."""
    tags = ["O"] * len(tokens)
    for entity in entities:
        entity_len = entity.end_idx - entity.start_idx
        tags[entity.start_idx : entity.end_idx] = [f"B-{entity.tag}"] + [
            f"I-{entity.tag}"
        ] * (entity_len - 1)
    return tags


def remove_correct_entity(
    tokens: list[str], pred: list[NEREntity], ref: list[NEREntity]
) -> list[NEREntity] | None:
    right_preds = set(ref).intersection(set(pred))
    if len(right_preds) == 0:
        return None

    new_pred = deepcopy(pred)
    new_pred.remove(random.choice(list(right_preds)))
    return new_pred


def add_wrong_entity(
    tokens: list[str], pred: list[NEREntity], ref: list[NEREntity], max_size: int = 3
) -> list[NEREntity] | None:
    def is_overlapping(span1: tuple[int, int], span2: tuple[int, int]) -> bool:
        return len(set(range(*span1)).intersection(set(range(*span2)))) > 0

    pred_spans = [(entity.start_idx, entity.end_idx) for entity in pred]
    ref_spans = [(entity.start_idx, entity.end_idx) for entity in ref]

    # all span of size up to max_size
    wrong_spans = {(i, j) for i, j in spans_indexs(tokens, max_len=max_size)}
    # filter overlapping entities
    wrong_spans = {
        span
        for span in wrong_spans
        if not any(is_overlapping(span, pred_span) for pred_span in pred_spans)
    }
    # filter right entities
    wrong_spans = wrong_spans - set(ref_spans)
    if len(wrong_spans) == 0:
        return None

    new_pred = deepcopy(pred)
    wrong_span = random.choice(list(wrong_spans))
    wrong_tag = random.choice(["PER", "LOC", "ORG"])
    new_pred.append(
        NEREntity(
            tokens[wrong_span[0] : wrong_span[1]],
            wrong_span[0],
            wrong_span[1],
            wrong_tag,
        )
    )
    return new_pred


def deteriorate_ner(
    tokens: list[str],
    pred: list[NEREntity],
    ref: list[NEREntity],
    actions: Optional[list[Callable]] = None,
) -> list[NEREntity] | None:
    """
    :param tokens: document tokens
    :param pred: list of predicted entities
    :param ref: list of reference entities
    :param actions: a list of deterioration actions.  None defaults to
        [remove_right_entity, add_wrong_entity]
    """
    actions = actions or [
        remove_correct_entity,
        add_wrong_entity,
    ]

    results = [action(tokens, pred, ref) for action in actions]
    results = [r for r in results if not r is None]
    if len(results) == 0:
        return None
    return random.choice(results)
