from typing import List, Dict, Optional
import numpy as np
import networkx as nx
from renard.pipeline.character_unification import Character


def shared_layout(
    G: nx.Graph,
    H_list: List[nx.Graph],
    H_to_G_list: List[Dict[Character, Optional[Character]]],
) -> Dict[Character, np.ndarray]:
    """Compute a layout shared between a reference graph G and a list
    of related graphs H.

    :param G: a reference graph with Character nodes
    :param H_list: a list of related graphs with Character nodes
    :param H_to_G_list: A list of mapping from V_H to V_G ∪ {∅}

    :return: a dict of form ``{character: (x, y)}``
    """
    # Extract the union of both graph nodes, considering the mapping
    # between G and H.
    nodes = list(G.nodes)
    for H, H_to_G in zip(H_list, H_to_G_list):
        for H_node, G_node in H_to_G.items():
            if G_node is None:
                nodes.append(H_node)

    # We do something similar to above here, except we define the
    # following mapping g_E from E_H to E_G:
    # { (g_V(n1), g_V(n2)) if g_V(n1) ≠ ∅ and g_V(n2) ≠ ∅,
    # { ∅                  otherwise
    edges = list(G.edges)
    for H, H_to_G in zip(H_list, H_to_G_list):
        for n1, n2 in H.edges:
            if H_to_G[n1] is None or H_to_G[n2] is None:
                n1 = H_to_G.get(n1) or n1
                n2 = H_to_G.get(n2) or n2
                edges.append((n1, n2))

    # Construct the union graph J between G and H
    J = nx.Graph()
    for node in nodes:
        J.add_node(node)
    for edge in edges:
        J.add_edge(*edge)

    # union graph layout
    # layout = layout_nx_graph_reasonably(J)
    layout = nx.kamada_kawai_layout(J)

    # the layout will be used for both G and all graphs in
    # H_list. However, some nodes from these H are not in the layout
    # dictionary: only their equivalent in G are here. We add these
    # nodes now, by specifying that their positions is the same as the
    # position of their equivalent nodes in G
    for H, H_to_G in zip(H_list, H_to_G_list):
        for node in H.nodes:
            if not node in layout:
                layout[node] = layout[H_to_G[node]]

    return layout
