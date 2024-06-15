from __future__ import annotations
import pytest
import networkx as nx
from renard.pipeline.character_unification import Character
from splice.metrics import (
    align_characters,
    score_character_unification,
    score_network_extraction_edges,
)


@pytest.mark.parametrize(
    "ref,pred,expected",
    [
        ([{"Elric"}], [{"Elric"}], (1.0, 1.0, 1.0)),
        ([{"Elric"}, {"Cymoril"}], [{"Elric"}], (1.0, 0.5, 0.667)),
        ([{"Elric"}], [{"Elric"}, {"Cymoril"}], (0.5, 1.0, 0.667)),
    ],
)
def test_score_characters_unification_canonical_examples(
    ref: list[set[str]], pred: list[set[str]], expected: tuple[float, float, float]
):
    precision, recall, f1 = score_character_unification(ref, pred)
    assert expected[0] == "*" or precision == pytest.approx(expected[0], rel=1e-2)
    assert expected[1] == "*" or recall == pytest.approx(expected[1], rel=1e-2)
    assert expected[2] == "*" or f1 == pytest.approx(expected[2], rel=1e-2)


@pytest.mark.parametrize(
    "ref,pred,expected",
    [
        ([("Elric", "Cymoril")], [("Elric", "Cymoril")], (1.0, 1.0, 1.0)),
        (
            [("Elric", "Cymoril"), ("Elric", "Yyrkoon")],
            [("Elric", "Cymoril")],
            (1.0, 0.5, 0.667),
        ),
        (
            [("Elric", "Cymoril")],
            [("Elric", "Cymoril"), ("Elric", "Yyrkoon")],
            (0.5, 1.0, 0.667),
        ),
    ],
)
def test_score_edge_extraction_canonical_examples(
    ref: list[tuple[str, str]],
    pred: list[tuple[str, str]],
    expected: tuple[float, float, float],
):
    G = nx.Graph()
    for a, b in ref:
        G.add_edge(Character(frozenset([a]), []), Character(frozenset([b]), []))

    P = nx.Graph()
    for a, b in pred:
        P.add_edge(Character(frozenset([a]), []), Character(frozenset([b]), []))

    mapping, _ = align_characters(list(G.nodes), list(P.nodes))
    precision, recall, f1 = score_network_extraction_edges(G, P, mapping)
    assert expected[0] == "*" or precision == pytest.approx(expected[0], rel=1e-2)
    assert expected[1] == "*" or recall == pytest.approx(expected[1], rel=1e-2)
    assert expected[2] == "*" or f1 == pytest.approx(expected[2], rel=1e-2)


@pytest.mark.parametrize(
    "ref,pred,expected",
    [
        ([("Elric", "Cymoril", 2)], [("Elric", "Cymoril", 1)], (0.5, 0.5, 0.5)),
        (
            [("Elric", "Cymoril", 2), ("Elric", "Yyrkoon", 1)],
            [("Elric", "Cymoril", 1)],
            (0.5, 0.25, "*"),
        ),
        (
            [("Elric", "Cymoril", 1)],
            [("Elric", "Cymoril", 2), ("Elric", "Yyrkoon", 1)],
            (0.25, 0.5, "*"),
        ),
    ],
)
def test_score_weighted_edge_extraction_canonical_examples(
    ref: list[tuple[str, str, float]],
    pred: list[tuple[str, str, float]],
    expected: tuple[float, float, float],
):
    G = nx.Graph()
    for a, b, w in ref:
        G.add_edge(
            Character(frozenset([a]), []), Character(frozenset([b]), []), weight=w
        )

    P = nx.Graph()
    for a, b, w in pred:
        P.add_edge(
            Character(frozenset([a]), []), Character(frozenset([b]), []), weight=w
        )

    mapping, _ = align_characters(list(G.nodes), list(P.nodes))
    precision, recall, f1 = score_network_extraction_edges(G, P, mapping, weighted=True)
    assert expected[0] == "*" or precision == pytest.approx(expected[0], rel=1e-2)
    assert expected[1] == "*" or recall == pytest.approx(expected[1], rel=1e-2)
    assert expected[2] == "*" or f1 == pytest.approx(expected[2], rel=1e-2)
