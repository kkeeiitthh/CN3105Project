"""ESOL data processing. Use from a notebook:

    from ex3_data_processing import get_molecules
    sets = get_molecules()          # {"train": Molecules, "val": Molecules, "test": Molecules}
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from ex2_graph_helper import smiles_to_graph

RDLogger.DisableLog("rdApp.*")

# ---- config: edit these -----------------------------------------------------
HERE     = Path(__file__).resolve().parent      # paths are relative to this file, not the notebook's cwd
DATA_DIR = HERE / "../data/Example 2 - ESOL"                # ESoL_train.csv / ESoL_val.csv / ESoL_test.csv (scaffold split)
TARGET   = "measured log solubility in mols per litre"
ESOL     = "ESOL predicted log solubility in mols per litre"   # Delaney's linear equation, used as a reference only
# ------------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def _check_scaffold_split(dfs):
    """The files are split by Bemis-Murcko scaffold: no ring system may appear in two splits."""
    scaf = {k: {MurckoScaffold.MurckoScaffoldSmiles(s) for s in d.smiles} for k, d in dfs.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (scaf[a] & scaf[b]), f"{a} and {b} share a scaffold"


class Molecules:
    """One split. Each molecule is a PyG Data with:
         x          (n_atoms, 177)   one-hot atom features
         edge_index (2, n_bonds * 2) bonds, both directions
         y          (1,)             standardised log S      (target)
    Targets are standardised with the mean/std of the TRAINING set (`mu`, `sd`).
    """

    def __init__(self, df, mu, sd):
        self.mu, self.sd = float(mu), float(sd)
        self.smiles = [s.strip() for s in df["smiles"]]
        self.y_raw = df[TARGET].to_numpy(np.float32)             # log S in mol/L
        self.esol = df[ESOL].to_numpy(np.float32)
        self.graphs = []
        for smi, y in zip(self.smiles, self.y_raw):
            g = smiles_to_graph(smi)
            g.y = torch.tensor([(y - self.mu) / self.sd], dtype=torch.float32)
            self.graphs.append(g)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]

    def unscale(self, p):
        """Standardised predictions -> log S."""
        return np.asarray(p) * self.sd + self.mu


def get_molecules():
    """Return {"train": Molecules, "val": Molecules, "test": Molecules}.

    The target is standardised with the training-set mean/std only.
    """
    dfs = {k: pd.read_csv(DATA_DIR / f"ESoL_{k}.csv") for k in SPLITS}
    _check_scaffold_split(dfs)
    y_train = dfs["train"][TARGET].to_numpy(np.float32)
    return {k: Molecules(d, y_train.mean(), y_train.std()) for k, d in dfs.items()}
