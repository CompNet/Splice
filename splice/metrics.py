import math
from typing import List, Literal, Tuple, Dict, Optional, Set, cast
from renard.pipeline.character_unification import Character
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from seqeval.metrics import precision_score, recall_score, f1_score
from renard.pipeline.core import Mention
from renard.pipeline.ner import NEREntity
from tibert.bertcoref import CoreferenceDocument
from tibert.score import score_coref_predictions, score_mention_detection
from splice.ner import entities_to_BIO


def align_characters(
    refs: List[Character], preds: List[Character]
) -> Tuple[Dict[Character, Optional[Character]], Dict[Character, Optional[Character]]]:
    """Try to best align a set of predicted characters to a list of
    reference characters.

    :return: a tuple:

            - a dict with keys from ``refs`` and values from
              ``preds``.  Values can be ``None``.

            - a dict with keys from ``preds`` and values from
              ``refs``.  Values can be ``None``.
    """
    similarity = np.zeros((len(refs), len(preds)))
    for r_i, ref_character in enumerate(refs):
        for p_i, pred_character in enumerate(preds):
            intersection = ref_character.names.intersection(pred_character.names)
            union = ref_character.names.union(pred_character.names)
            similarity[r_i][p_i] = len(intersection) / len(union)

    graph = csr_matrix(similarity)
    perm = maximum_bipartite_matching(graph, perm_type="column")

    mapping = [[char, None] for char in refs]
    for r_i, mapping_i in enumerate(perm):
        if perm[r_i] != -1:
            mapping[r_i][1] = preds[mapping_i]

    mapping = {c1: c2 for c1, c2 in mapping}

    reverse_mapping = {v: k for k, v in mapping.items()}
    for character in preds:
        if not character in reverse_mapping:
            reverse_mapping[character] = None

    return (mapping, reverse_mapping)


def score_network_extraction_edges(
    gold_graph: nx.Graph,
    pred_graph: nx.Graph,
    mapping: Dict[Character, Optional[Character]],
    weighted: bool = False,
) -> Tuple[float, float, float]:
    """
    :return: precision, recall, f1
    """
    epsilon = 1e-9
    max_ref_weight = max(
        [gold_graph.edges[edge].get("weight", 0) for edge in gold_graph.edges],
        default=epsilon,
    )
    max_pred_weight = max(
        [pred_graph.edges[edge].get("weight", 0) for edge in pred_graph.edges],
        default=epsilon,
    )

    recall_list = []
    for r1, r2 in gold_graph.edges:
        c1 = mapping[r1]
        c2 = mapping[r2]
        if (c1, c2) in pred_graph.edges:
            sim = 1
            if weighted:
                ref_weight = gold_graph.edges[(r1, r2)]["weight"] / max_ref_weight
                pred_weight = pred_graph.edges[(c1, c2)]["weight"] / max_pred_weight
                sim = min(1, 1 - abs(ref_weight - pred_weight))
            recall_list.append(sim)
        else:
            recall_list.append(0)
    if len(recall_list) == 0:
        recall = float("NaN")
    else:
        recall = sum(recall_list) / len(recall_list)

    # edge precision
    precision_list = []
    reverse_mapping = {v: k for k, v in mapping.items() if not v is None}
    for c1, c2 in pred_graph.edges:
        r1 = reverse_mapping.get(c1)
        r2 = reverse_mapping.get(c2)
        if (r1, r2) in gold_graph.edges:
            sim = 1
            if weighted:
                ref_weight = gold_graph.edges[(r1, r2)]["weight"] / max_ref_weight
                pred_weight = pred_graph.edges[(c1, c2)]["weight"] / max_pred_weight
                sim = min(1, 1 - abs(ref_weight - pred_weight))
            precision_list.append(sim)
        else:
            precision_list.append(0)
    if len(precision_list) == 0:
        precision = float("NaN")
    else:
        precision = sum(precision_list) / len(precision_list)

    if precision + recall == 0:
        return (precision, recall, 0)

    f1 = float("NaN")
    if not math.isnan(precision) and not math.isnan(recall) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def score_character_unification(
    refs: List[Set[str]], preds: List[Set[str]]
) -> Tuple[float, float, float]:
    """Score the character unification task.

    >>> score_character_unification([{"Arthur", "Amalvy"}], [{"Arthur", "Amalvy"}])
    (1.0, 1.0, 1.0)

    :return: (precision, recall, f1)
    """
    preds_nb = len(preds)
    refs_nb = len(refs)

    precision_scores = np.zeros((preds_nb, refs_nb))
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            precision_scores[p_i][r_i] = 1 - len(pred - ref) / len(pred)
    graph = csr_matrix(precision_scores)
    perm = maximum_bipartite_matching(graph, perm_type="column")
    # sum only the pair of name sets that maps together
    mapped_precision_scores = np.take_along_axis(precision_scores, perm[:, None], 1)
    mapped_precision_scores = mapped_precision_scores.flatten()
    # ignore unmapped sets
    mapped_precision_scores *= (perm != -1).astype(int)
    if preds_nb == 0:
        precision = float("NaN")
    else:
        precision = sum(mapped_precision_scores) / preds_nb

    recall_scores = np.zeros((preds_nb, refs_nb))
    for r_i, ref in enumerate(refs):
        for p_i, pred in enumerate(preds):
            recall_scores[p_i][r_i] = 1 if len(ref.intersection(pred)) > 0 else 0
    graph = csr_matrix(recall_scores)
    perm = maximum_bipartite_matching(graph, perm_type="column")
    # sum only the pair of name sets that maps together
    mapped_recall_scores = np.take_along_axis(recall_scores, perm[:, None], 1)
    mapped_recall_scores = mapped_recall_scores.flatten()
    # ignore unmapped sets
    mapped_recall_scores *= (perm != -1).astype(int)
    if refs_nb == 0:
        recall = float("NaN")
    else:
        recall = sum(mapped_recall_scores) / refs_nb

    f1 = float("NaN")
    if not math.isnan(precision) and not math.isnan(recall) and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def score_ner(
    tokens: List[str],
    pred_entities: List[NEREntity],
    ref_entities: List[NEREntity],
    ignore_classes: Optional[Set[str]],
) -> Tuple[float, float, float]:
    """Score NER prediction against a reference.

    :return: (precision, recall, f1)
    """
    if not ignore_classes is None:
        pred_entities = [ent for ent in pred_entities if not ent.tag in ignore_classes]
        ref_entities = [ent for ent in ref_entities if not ent.tag in ignore_classes]

    pred_tags = entities_to_BIO(tokens, pred_entities)
    ref_tags = entities_to_BIO(tokens, ref_entities)

    return (
        precision_score([ref_tags], [pred_tags]),
        recall_score([ref_tags], [pred_tags]),
        f1_score([ref_tags], [pred_tags]),
    )


def score_coref(
    tokens: List[str], pred_corefs: List[List[Mention]], ref_corefs: List[List[Mention]]
) -> Dict[str, Dict[Literal["precision", "recall", "f1"], float]]:
    pred_doc = CoreferenceDocument(tokens, pred_corefs)
    ref_doc = CoreferenceDocument(tokens, ref_corefs)

    scores = score_coref_predictions([pred_doc], [ref_doc])
    scores = cast(Dict[str, Dict[Literal["precision", "recall", "f1"], float]], scores)

    precision, recall, f1 = score_mention_detection([pred_doc], [ref_doc])
    scores["mention_detection"] = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return scores
