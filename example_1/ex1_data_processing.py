"""TEP data processing. Use from a notebook:

    from ex1_data_processing import get_windows
    sets = get_windows()          # {"train": Windows, "val": Windows, "test": Windows}
"""
import numpy as np
import pandas as pd
from pathlib import Path


# ---- config: edit these -----------------------------------------------------
HERE     = Path(__file__).resolve().parent      # paths are relative to this file, not the notebook's cwd
PQ_DIR   = HERE / "../data/TEP"
CACHE_DIR = HERE / "../cache"
FILES = {
    ("train", "free"):   "FaultFree_Training.csv",
    ("train", "faulty"): "Faulty_Training.csv",
    ("test",  "free"):   "FaultFree_Testing.csv",
    ("test",  "faulty"): "Faulty_Testing.csv",
}
FAULTS  = [1, 4, 8, 13, 14]    # fault classes to keep
N_TRAIN = 50                   # runs per class used for training
N_VAL   = 10                   # further runs per class, held out for validation
N_TEST  = 50                   # runs per class taken from the test files
W, STRIDE = 30, 10             # window length and stride, in rows (3 min each)
ONSET = {"train": 20, "val": 20, "test": 160}   # fault-free rows at the start of faulty runs; verify on your copy
# ------------------------------------------------------------------------------

META = ["faultNumber", "simulationRun", "sample"]
FEATURES = [f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)]  # 52 variables
CLASSES = [0] + sorted(FAULTS)                       # labels are indices into this list

# graph nodes: xmeas_1-22, xmv_1-11 (xmeas_23-41 dropped); node i (1-indexed) = ORDER[i-1]
N_NODES = 33
_col = {n: i for i, n in enumerate(FEATURES)}
ORDER = [_col[f"xmeas_{i}"] for i in range(1, 23)] + [_col[f"xmv_{i}"] for i in range(1, 12)]
NODE_NAMES = [FEATURES[i] for i in ORDER]
assert len(ORDER) == N_NODES and len(set(ORDER)) == N_NODES


def _read_filtered(kind, which, faults, max_run):
    """Parquet only reads the columns and row groups that can match the filter."""
    path = PQ_DIR / FILES[(kind, which)].replace(".csv", ".parquet")
    df = pd.read_parquet(path, columns=META + FEATURES,
                         filters=[("faultNumber", "in", faults), ("simulationRun", "<=", max_run)])
    return df.sort_values(META, ignore_index=True)


def _to_runs(df):
    """Long table -> array of shape (runs, time, 52), plus the fault number and run id of each run."""
    sizes = df.groupby(["faultNumber", "simulationRun"], sort=True).size()
    T = int(sizes.iloc[0])
    assert (sizes == T).all(), "runs have different lengths"
    sample = df["sample"].to_numpy().reshape(-1, T)
    assert (sample == np.arange(1, T + 1)).all(), "sample is not 1..T inside every run"
    X = df[FEATURES].to_numpy(np.float32).reshape(-1, T, len(FEATURES))
    keys = sizes.index.to_frame(index=False)
    return X, keys["faultNumber"].to_numpy(np.int16), keys["simulationRun"].to_numpy(np.int16)


def _load_kind(kind, max_run):
    free = _to_runs(_read_filtered(kind, "free", [0], max_run))
    faulty = _to_runs(_read_filtered(kind, "faulty", FAULTS, max_run))
    return tuple(np.concatenate(pair) for pair in zip(free, faulty))


def _select(X, fault, run, lo, hi):
    m = (run > lo) & (run <= hi)
    return {"X": X[m], "fault": fault[m], "run": run[m]}


def _load_data():
    """Raw arrays for train / val / test, cached as .npz after the first build."""
    tag = f"f{'-'.join(map(str, FAULTS))}_tr{N_TRAIN}_va{N_VAL}_te{N_TEST}"
    cache = CACHE_DIR / f"tep_{tag}.npz"
    CACHE_DIR.mkdir(exist_ok=True)

    if cache.exists():
        z = np.load(cache)
        return {s: {k: z[f"{s}_{k}"] for k in ("X", "fault", "run")} for s in ("train", "val", "test")}

    Xa, fa, ra = _load_kind("train", N_TRAIN + N_VAL)
    Xb, fb, rb = _load_kind("test", N_TEST)
    data = {
        "train": _select(Xa, fa, ra, 0, N_TRAIN),
        "val":   _select(Xa, fa, ra, N_TRAIN, N_TRAIN + N_VAL),   # held-out runs, same files
        "test":  _select(Xb, fb, rb, 0, N_TEST),
    }
    np.savez(cache, **{f"{s}_{k}": v for s, d in data.items() for k, v in d.items()})
    return data


class Windows:
    """Index of (run, end-row) pairs over standardised runs; each window is sliced out on demand."""

    def __init__(self, d, onset, mu, sd, w=W, stride=STRIDE):
        self.w = w
        self.X = ((d["X"] - mu) / sd)[..., ORDER]    # float32, shape (runs, time, 33), node order
        R, T, _ = self.X.shape
        grid = np.meshgrid(np.arange(R), np.arange(w - 1, T, stride), indexing="ij")
        run_i, end_i = (g.ravel() for g in grid)
        fault = d["fault"][run_i]
        pre = (fault == 0) | (end_i < onset)         # fault-free run, or window ends before onset
        post = (fault > 0) & (end_i - w + 1 >= onset)  # window lies entirely after onset
        keep = pre | post                            # windows straddling onset are dropped
        self.run_i, self.end_i = run_i[keep], end_i[keep]
        self.y = np.searchsorted(CLASSES, np.where(post, fault, 0)[keep]).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        r, e = self.run_i[i], self.end_i[i]
        return self.X[r, e - self.w + 1: e + 1].T, self.y[i]   # (33 nodes, w), class index

    def batch(self, idx):
        return np.stack([self[i][0] for i in idx]), self.y[idx]  # (B, 33, w), (B,)


def get_windows(w=W, stride=STRIDE):
    """Return {"train": Windows, "val": Windows, "test": Windows}.

    Standardisation uses per-variable mean/std from fault-free TRAINING runs only.
    """
    data = _load_data()
    train = data["train"]
    normal = train["X"][train["fault"] == 0].reshape(-1, train["X"].shape[-1])
    mu = normal.mean(0).astype(np.float32)
    sd = normal.std(0).astype(np.float32)
    sd[sd == 0] = 1.0
    return {s: Windows(data[s], ONSET[s], mu, sd, w, stride) for s in ("train", "val", "test")}
