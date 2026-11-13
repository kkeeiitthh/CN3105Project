"""Molecule -> graph helpers for the GCN. Use from a notebook:

    from ex3_graph_helper import smiles_to_graph, plot_molecule_graph, N_NODE_FEATURES
    g = smiles_to_graph("CCO")      # Data(x=[3, 177], edge_index=[2, 4])
"""
import io
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles
from torch_geometric.utils.smiles import x_map

# from_smiles codes each atom as 9 integers (atomic number, chirality, degree, formal charge, number of Hs,
# radical electrons, hybridisation, aromatic, in-ring). GCNConv multiplies x by a weight matrix, so the integer
# codes (categories, not magnitudes) are expanded to one-hot vectors and concatenated: 119+9+11+12+9+5+8+2+2 = 177.
SIZES = [len(v) for v in x_map.values()]
OFFSETS = torch.tensor(np.cumsum([0] + SIZES[:-1]))
N_NODE_FEATURES = int(sum(SIZES))

ELEMENT_COLOURS = {"C": "lightgray", "O": "lightcoral", "N": "lightblue", "S": "khaki",
                   "Cl": "lightgreen", "F": "palegreen", "Br": "sandybrown", "P": "plum", "I": "orchid"}


def onehot_atoms(x_int):
    """(N, 9) integer codes -> (N, 177) float one-hot."""
    n = x_int.shape[0]
    x = torch.zeros(n, N_NODE_FEATURES)
    x[torch.arange(n)[:, None], x_int + OFFSETS] = 1.0
    return x


def smiles_to_graph(smiles):
    """SMILES -> Data(x, edge_index). Edges are stored in both directions (undirected).

    from_smiles also returns bond features (edge_attr); they are dropped here because GCNConv has no
    way to use them. Only the bond *pattern* (edge_index) is kept.
    """
    g = from_smiles(smiles)
    return Data(x=onehot_atoms(g.x), edge_index=g.edge_index)


def plot_molecule_graph(smiles, save_as="molecule_graph.png"):
    """Left: structure with atom indices. Right: the graph the GCN sees (nodes = atoms, edges = bonds)."""
    mol = Chem.MolFromSmiles(smiles)
    g = smiles_to_graph(smiles)

    drawer = rdMolDraw2D.MolDraw2DCairo(480, 380)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = plt.imread(io.BytesIO(drawer.GetDrawingText()), format="png")

    G = nx.Graph()                                   # undirected: the two stored directions collapse to one edge
    G.add_nodes_from(range(mol.GetNumAtoms()))
    G.add_edges_from(g.edge_index.T.tolist())
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = nx.kamada_kawai_layout(G) if len(G) > 2 else nx.spring_layout(G, seed=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white")
    ax[0].imshow(img); ax[0].axis("off"); ax[0].set_title(smiles)
    nx.draw_networkx(G, pos, ax=ax[1], labels={i: f"{s}{i}" for i, s in enumerate(sym)},
                     node_color=[ELEMENT_COLOURS.get(s, "white") for s in sym], edgecolors="k",
                     node_size=650, font_size=9, edge_color="gray", width=1.2)
    ax[1].axis("off")
    ax[1].set_title(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} bonds "
                    f"({g.edge_index.shape[1]} directed edges)")
    plt.tight_layout()
    plt.savefig(save_as, dpi=200, facecolor="white", bbox_inches="tight")   # before show(), which closes the figure
    plt.show()
