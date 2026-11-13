import itertools
import networkx as nx
import matplotlib.pyplot as plt
import torch
from ex1_data_processing import N_NODES

def build_graph(edges):
    G = nx.Graph()          # undirected, consistent with the edge_index construction
    G.add_edges_from(edges)
    return G

def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def count_crossings(G, pos):
    n = 0
    for (a, b), (c, d) in itertools.combinations(G.edges, 2):
        if len({a, b, c, d}) < 4:       # edges sharing a node can't cross
            continue
        A, B, C, D = pos[a], pos[b], pos[c], pos[d]
        if _ccw(A, C, D) != _ccw(B, C, D) and _ccw(A, B, C) != _ccw(A, B, D):
            n += 1
    return n

def neat_layout(G, tries=100):
    # Kamada-Kawai from many random starts; keep the one with fewest edge crossings
    layouts = (nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=s)) for s in range(tries))
    return min(layouts, key=lambda p: count_crossings(G, p))



def make_edge_index(edges):
    ei = torch.tensor(edges).T - 1                       # 0-indexed
    return torch.cat([ei, ei.flip(0)], dim=1)            # undirected


def get_edge_index(adjacency_list, node_names=None, plot=True, verbose=True):
    """Undirected graph as a [2, E] long tensor (0-indexed), the form GCNConv takes.

    plot=True also draws (and saves) the graph; node_names maps 1-indexed node -> label.
    """
    edge_index = make_edge_index(adjacency_list)
    assert edge_index.max() == N_NODES - 1
    if verbose:
        print("graph:", N_NODES, "nodes,", edge_index.shape[1], "directed edges")
    if plot:
        plot_graph(adjacency_list, node_names or {})
    return edge_index


def plot_graph(adjacency_list, node_names):
    g = build_graph(adjacency_list)
    pos = neat_layout(g)

    # Flatten: stretch x relative to y (scaling keeps edges straight, so crossings are unchanged)
    STRETCH = 2.5
    pos = {n: (x * STRETCH, y) for n, (x, y) in pos.items()}

    node_color = ["lightblue" if n >= 23 else "lightcoral" for n in g.nodes]

    fig, ax = plt.subplots(figsize=(16, 6.5), facecolor="white")
    ax.set_facecolor("white")
    nx.draw_networkx(
        g, pos, ax=ax,
        labels={n: node_names.get(n, str(n)) for n in g.nodes},
        node_color=node_color, edgecolors="k", node_size=450,
        font_size=9, edge_color="gray", width=1.0,
    )

    ax.set_title(f"adjacency_list_actuators\n{g.number_of_nodes()} nodes, {g.number_of_edges()} edges, {count_crossings(g, pos)} crossings", color="black")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("graph_actuators.png", dpi=200, facecolor="white", bbox_inches="tight")  # before show(), which closes the figure
    plt.show()
