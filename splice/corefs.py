from __future__ import annotations
from typing import Callable, Optional, Generator
import random
from copy import deepcopy
import itertools as it
import functools as ft
from more_itertools import flatten
from sacred.run import Run
from tibert.bertcoref import (
    CoreferenceDocument,
    CoreferenceDataset,
    Mention,
    BertForCoreferenceResolution,
)
from tibert.train import train_coref_model as _train_coref_model
from transformers import BertTokenizerFast
from tibert.utils import spans_indexs


@ft.lru_cache(maxsize=None)
def _cached_spans_indexs(seq: tuple, max_len: int) -> list[tuple[int, int]]:
    return spans_indexs(seq, max_len)  # type: ignore


def coref_links(doc: CoreferenceDocument) -> set[tuple[Mention, Mention]]:
    """
    Given the current chains, returns all the existing coreference
    links.
    """
    links = []
    for chain in doc.coref_chains:
        sorted_chain = sorted(chain, key=lambda mention: mention.start_idx)
        links += list(it.combinations(sorted_chain, 2))
    return set(links)


def possible_coref_links(
    doc: CoreferenceDocument,
) -> Generator[tuple[Mention, Mention], None, None]:
    """
    Given the current chains, returns all the possible additional
    links that could exist.
    """
    if len(doc.coref_chains) <= 1:
        return

    links = []
    for chain1, chain2 in it.combinations(doc.coref_chains, 2):
        links += list(it.product(chain1, chain2))

    while len(links) > 0:
        i = random.randrange(len(links))
        links[i], links[-1] = links[-1], links[i]
        yield links.pop()


def add_wrong_mention(
    pred: CoreferenceDocument, ref: CoreferenceDocument, max_span_size: int
) -> CoreferenceDocument | None:
    ret = deepcopy(pred)
    possible_spans = {
        (s, e) for s, e in _cached_spans_indexs(tuple(pred.tokens), max_span_size)
    }
    ref_mentions = {(m.start_idx, m.end_idx) for m in flatten(ref.coref_chains)}
    possible_spans -= ref_mentions
    if len(possible_spans) == 0:
        return None

    wrong_mention = random.choice(list(possible_spans))
    ret.coref_chains.append(
        [
            Mention(
                pred.tokens[wrong_mention[0] : wrong_mention[1]],
                wrong_mention[0],
                wrong_mention[1],
            )
        ]
    )

    return ret


def remove_correct_mention(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    """Remove a correct mention from pred"""

    ref_mentions = set(flatten(ref.coref_chains))
    pred_mentions = set(flatten(pred.coref_chains))
    correct_mentions = pred_mentions & ref_mentions
    if len(correct_mentions) == 0:
        return None
    ret = deepcopy(pred)
    correct_mention = random.choice(list(correct_mentions))
    for chain in ret.coref_chains:
        try:
            chain.remove(correct_mention)
        except ValueError:
            pass
    ret.coref_chains = [c for c in ret.coref_chains if len(c) > 0]
    return ret


def add_wrong_link(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    """Add an incorrect coreference link to pred"""

    if len(pred.coref_chains) <= 1:
        return None

    ref_links = coref_links(ref)
    pred_links = coref_links(pred)
    U_ref_pred_links = ref_links.union(pred_links)

    wrong_link = None
    for link in possible_coref_links(pred):
        if link in U_ref_pred_links:
            continue
        wrong_link = link
        break

    if wrong_link is None:
        return None

    ret = CoreferenceDocument(pred.tokens, [])
    chain1 = next(c for c in pred.coref_chains if wrong_link[0] in c)
    chain2 = next(c for c in pred.coref_chains if wrong_link[1] in c)
    for chain in pred.coref_chains:
        if chain == chain1 or chain == chain2:
            continue
        ret.coref_chains.append(chain)
    ret.coref_chains.append(chain1 + chain2)

    return ret


def remove_correct_link(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    """Remove a correct coreference link from pred"""
    if len(pred.coref_chains) <= 1:
        return None

    ref_links = coref_links(ref)
    pred_links = coref_links(pred)
    correct_links = ref_links & pred_links
    if len(correct_links) == 0:
        return None

    ret = CoreferenceDocument(pred.tokens, [])
    correct_link = random.choice(list(correct_links))
    affected_chain = next(c for c in pred.coref_chains if correct_link[0] in c)
    for chain in pred.coref_chains:
        if chain == affected_chain:
            split_point = random.randint(1, len(chain) - 1)
            ret.coref_chains.append(chain[:split_point])
            ret.coref_chains.append(chain[split_point:])
        else:
            ret.coref_chains.append(chain)
    ret.coref_chains = [c for c in ret.coref_chains if len(c) > 0]
    return ret


def deteriorate_coref(
    pred: CoreferenceDocument,
    ref: CoreferenceDocument,
    actions: Optional[list[Callable]] = None,
) -> CoreferenceDocument | None:
    """Randomly deteriorate coreference prediction

    :param pred: coreference prediction
    :param ref: coreference reference
    :param actions: a list of deterioration actions.  None defaults to
        [ft.partial(add_wrong_mention, max_span_size=10),
        remove_correct_mention, add_wrong_link, remove_correct_link]
    """
    assert pred.tokens == ref.tokens

    # We find 4 possible ways to degrade a coref prediction
    actions = actions or [
        ft.partial(add_wrong_mention, max_span_size=10),
        remove_correct_mention,
        add_wrong_link,
        remove_correct_link,
    ]
    # Then, we simply sample an action between these and return it. If
    # no action is possible, return None
    results = [action(pred, ref) for action in actions]
    results = [r for r in results if not r is None]
    if len(results) == 0:
        return None
    return random.choice(results)


def add_correct_mention(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    right_mentions = set(flatten(ref.coref_chains))
    pred_mentions = set(flatten(pred.coref_chains))
    mentions_to_add = right_mentions - pred_mentions

    if len(mentions_to_add) == 0:
        return

    ret = deepcopy(pred)
    ret.coref_chains.append([random.choice(list(mentions_to_add))])
    return ret


def remove_wrong_mention(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    right_mentions = set(flatten(ref.coref_chains))
    pred_mentions = set(flatten(pred.coref_chains))
    wrong_mentions = pred_mentions - right_mentions

    if len(wrong_mentions) == 0:
        return None

    ret = CoreferenceDocument(pred.tokens, [])
    wrong_mention = random.choice(list(wrong_mentions))
    for chain in pred.coref_chains:
        chain = [m for m in chain if not m == wrong_mention]
        if len(chain) > 0:
            ret.coref_chains.append(chain)

    return ret


def add_correct_link(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    if len(pred.coref_chains) <= 1:
        return None

    ref_links = coref_links(ref)
    possible_pred_links = set(possible_coref_links(pred))
    possible_right_links = possible_pred_links & ref_links
    if len(possible_right_links) == 0:
        return None

    ret = CoreferenceDocument(pred.tokens, [])
    right_link = random.choice(list(possible_right_links))
    chain1 = next(c for c in pred.coref_chains if right_link[0] in c)
    chain2 = next(c for c in pred.coref_chains if right_link[1] in c)
    for chain in pred.coref_chains:
        if chain == chain1 or chain == chain2:
            continue
        ret.coref_chains.append(chain)
    ret.coref_chains.append(chain1 + chain2)

    return ret


def remove_wrong_link(
    pred: CoreferenceDocument, ref: CoreferenceDocument
) -> CoreferenceDocument | None:
    if len(pred.coref_chains) <= 1:
        return None

    ref_links = coref_links(ref)
    pred_links = coref_links(pred)
    wrong_links = pred_links - ref_links
    if len(wrong_links) == 0:
        return None

    ret = CoreferenceDocument(pred.tokens, [])
    wrong_link = random.choice(list(wrong_links))
    affected_chain = next(c for c in pred.coref_chains if wrong_link[0] in c)
    for chain in pred.coref_chains:
        if chain == affected_chain:
            new_chain = deepcopy(chain)
            new_chain.remove(wrong_link[0])
            ret.coref_chains.append([wrong_link[0]])
            ret.coref_chains.append(new_chain)
        else:
            ret.coref_chains.append(chain)

    return ret


def enhance_coref(
    pred: CoreferenceDocument,
    ref: CoreferenceDocument,
    actions: Optional[list[Callable]] = None,
) -> CoreferenceDocument | None:
    """
    :param pred: coreference predictions
    :param ref: coreference reference
    :param actions: a list of enhanchement actions.  None defaults to
        [add_right_mention, remove_wrong_mention, add_right_link,
        remove_wrong_link]
    """
    assert pred.tokens == ref.tokens

    # We find 4 possible ways to enhance a coref prediction
    actions = actions or [
        add_correct_mention,
        remove_wrong_mention,
        add_correct_link,
        remove_wrong_link,
    ]

    # Then, we simply sample an action between these and return it. If
    # no action is possible, return None
    results = [action(pred, ref) for action in actions]
    results = [r for r in results if not r is None]
    if len(results) == 0:
        return None
    return random.choice(results)


def train_coref_model(
    _run: Run,
    tokenizer: BertTokenizerFast,
    train_dataset: CoreferenceDataset,
) -> BertForCoreferenceResolution:
    coref_model = BertForCoreferenceResolution.from_pretrained(
        "bert-base-cased",
        mentions_per_tokens=0.4,
        antecedents_nb=350,
        max_span_size=train_dataset.max_span_size,
        segment_size=128,
        mention_scorer_hidden_size=300,
        mention_scorer_dropout=0.2,
        hidden_dropout_prob=0.2,
        attention_probs_droupout_prob=0.2,
        mention_loss_coeff=0.1,
    )

    traind, evald = train_dataset.splitted(0.9)

    coref_model = _train_coref_model(
        coref_model,
        traind,
        evald,
        tokenizer,
        batch_size=1,
        epochs_nb=50,
        bert_lr=1e-5,
        task_lr=2e-4,
        model_save_dir=f"./coref_model_out_{_run._id}",
        device_str="cuda",
        _run=_run,
    )

    return coref_model
