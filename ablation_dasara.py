"""
run_ablations_gnode.py

Single-file ablation study for the GN-ODE models (companion to run_ablations.py
for GN-SDE). Every block has its own control, every row is trained under the same
reduced budget, and DeltaR2 is computed against that block's control (paired by seed).

  (i)   Dynamics terms        : w/o growth, w/o diffusion, w/o enforcement
  (ii)  Graph structure       : fully connected, attention edges, shuffled graph
  (iii) Temporal components   : w/o seasonal growth rate, w/o initial-state smoother
  (iv)  Latent enforcement    : w/o trend, w/o smoothness penalty,
                                w/o latent inference (observed L), w/o enforcement
  (v)   Sensitivity           : smoother length, seasonal amplitude, lambda_L

USAGE (run from the project root)
--------------------------------------------------------------------
    python src/run_ablations_gnode.py --dataset chicago_districts
    python src/run_ablations_gnode.py --dataset chicago_districts --group latent
    python src/run_ablations_gnode.py --dataset chicago_beats --seeds 42 43 44
    python src/run_ablations_gnode.py --dataset chicago_districts --epochs 20 --patience 5   # fast dev pass
    python src/run_ablations_gnode.py --dataset chicago_districts --table_only               # rebuild table only

Produces (per dataset):
    data/processed/{dataset}_ablation_gnode_summary.csv       one row per (config, seed)
    data/processed/{dataset}_ablation_gnode_table.csv         mean/std over seeds + dR2
    data/processed/{dataset}_ablation_gnode_table.tex         paper-ready LaTeX table
    data/processed/{dataset}_ablation_gnode_{name}_seed{s}_predictions.csv
Finished (config, seed) pairs are skipped on re-run; use --force to redo them.
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ── locate the model package (put its parent FIRST on sys.path) ─────
def _find_model_parent():
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent, here / "src", here.parent / "src"):
        if (cand / "model" / "gnode_dasara.py").exists():
            return cand
    sys.exit("Could not find model/gnode_dasara.py next to or above this script.")


sys.path.insert(0, str(_find_model_parent()))
from model.gnode import (GNODEBase, GNODEfc, GNODEspatial,   # noqa: E402
                                GNODEattn, GNODElatent)

# ─────────────────────────────────────────────────────────────────────
# SETTINGS (same values as the main training script)
# ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
LR          = 1e-3
LR_L        = 5e-3
LAMBDA_L    = 5e-3
GRAD_CLIP   = 1.0
BATCH_PAIRS = 8
ODE_STEP    = 0.25
HIST_DEFAULT = 4
EPOCHS      = 100       # reduced budget for the whole sweep (main results use 300)
PATIENCE    = 15

DATA = {
    "chicago_beats": {
        "crime": "data/processed/chicago_beats_crime_timeseries.csv",
        "arrest": "data/processed/chicago_beats_arrest_timeseries.csv",
        "order": "data/raw/graph_beat_order.csv",
        "adj_fc": "data/processed/chicago_beats_adjacencyfc.npy",
        "adj_sp": "data/processed/chicago_beats_adjacency_spatial.npy",
    },
    "chicago_districts": {
        "crime": "data/processed/chicago_districts_crime_timeseries.csv",
        "arrest": "data/processed/chicago_districts_arrest_timeseries.csv",
        "order": "data/raw/graph_district_order.csv",
        "adj_fc": "data/processed/chicago_districts_adjacencyfc.npy",
        "adj_sp": "data/processed/chicago_districts_adjacency_spatial.npy",
    },
    "nyc_precincts": {
        "crime": "data/processed/nyc_precincts_crime_timeseries.csv",
        "arrest": "data/processed/nyc_precincts_arrest_timeseries.csv",
        "order": "data/raw/graph_precinct_order.csv",
        "adj_fc": "data/processed/nyc_precincts_adjacencyfc.npy",
        "adj_sp": "data/processed/nyc_precincts_adjacency_spatial.npy",
    },
}

BLOCK_TITLES = {
    "i":     "(i) Dynamics-term ablations",
    "ii":    "(ii) Graph-structure ablations",
    "iii":   "(iii) Temporal-component ablations",
    "iv":    "(iv) Latent-enforcement ablations",
    "v_hist": "(v-a) Sensitivity: smoother length (hist)",
    "v_amp":  "(v-b) Sensitivity: seasonal amplitude",
    "v_lam":  "(v-c) Sensitivity: smoothness weight lambda_L (latent)",
}
BLOCK_ORDER = list(BLOCK_TITLES)


# ─────────────────────────────────────────────────────────────────────
# ABLATION-CAPABLE MODEL SUBCLASSES (built on your model classes)
# ─────────────────────────────────────────────────────────────────────
class _AblateMixin:
    """Switches for the three terms of the vector field:
         dC/dt = growth - diffusion - enforcement
    A disabled term is replaced by zeros; all parameters still exist."""

    def __init__(self, *args, use_growth=True, use_diffusion=True,
                 use_enforcement=True, **kw):
        super().__init__(*args, **kw)
        self.use_growth = bool(use_growth)
        self.use_diffusion = bool(use_diffusion)
        self.use_enforcement = bool(use_enforcement)

    def forward(self, t, C):
        C = C.view(-1)
        zero = torch.zeros_like(C)
        growth = self.alpha_t() * C * (1.0 - C) if self.use_growth else zero
        if self.use_diffusion:
            K = self.coupling()
            diffusion = K.sum(dim=1) * C - K @ C
        else:
            diffusion = zero
        enforce = (self.beta * self.enforcement() * C
                   if self.use_enforcement else zero)
        return growth - diffusion - enforce


class AblFC(_AblateMixin, GNODEfc):
    pass


class AblSpatial(_AblateMixin, GNODEspatial):
    pass


class AblAttn(_AblateMixin, GNODEattn):
    pass


class AblLatent(_AblateMixin, GNODElatent):
    """use_latent=False: L_net is bypassed and the OBSERVED arrest rate is used
    as L (same as the spatial model), isolating the value of latent inference."""

    def __init__(self, *args, use_latent=True, **kw):
        super().__init__(*args, **kw)
        self.use_latent = bool(use_latent)

    def enforcement(self):
        if not self.use_latent:
            return GNODEBase.enforcement(self)
        return super().enforcement()


CLS = {"fc": AblFC, "spatial": AblSpatial, "spatial_attn": AblAttn, "latent": AblLatent}


# ─────────────────────────────────────────────────────────────────────
# ABLATION CONFIGS
# ─────────────────────────────────────────────────────────────────────
def build_configs(group):
    cfgs = []

    def add(block, key, label, variant, control=False, shuffle=False,
            lambda_L=LAMBDA_L, **mk):
        cfgs.append(dict(
            name=f"{block}_control" if control else f"{block}_{key}",
            block=block, label=label, role="control" if control else "ablation",
            variant=variant, adj="fc" if variant == "fc" else "sp",
            shuffle=shuffle, lambda_L=lambda_L, mk=mk))

    if group in ("dynamics", "all"):
        add("i", "control", "Control (full spatial)", "spatial", control=True)
        add("i", "no_growth", "w/o growth term", "spatial", use_growth=False)
        add("i", "no_diffusion", "w/o diffusion (K = 0)", "spatial", use_diffusion=False)
        add("i", "no_enforcement", "w/o enforcement", "spatial", use_enforcement=False)

    if group in ("graph", "all"):
        add("ii", "control", "Control (spatial contiguity)", "spatial", control=True)
        add("ii", "fc", "Fully connected graph", "fc")
        add("ii", "attn", "Attention-reweighted edges", "spatial_attn")
        add("ii", "shuffled", "Shuffled graph (geography destroyed)", "spatial", shuffle=True)

    if group in ("temporal", "all"):
        add("iii", "control", "Control (full spatial)", "spatial", control=True)
        add("iii", "no_season", "w/o seasonal growth rate", "spatial", seasonal=False)
        add("iii", "no_smoother", "w/o initial-state smoother (hist = 1)", "spatial", hist=1)

    if group in ("latent", "all"):
        add("iv", "control", "Control (full latent)", "latent", control=True)
        add("iv", "no_trend", "w/o trend feature", "latent", use_trend=False)
        add("iv", "no_smooth_pen", "w/o smoothness penalty", "latent", lambda_L=0.0)
        add("iv", "observed_L", "w/o latent inference (observed L)", "latent", use_latent=False)
        add("iv", "no_enforcement", "w/o enforcement", "latent", use_enforcement=False)

    if group in ("sensitivity", "all"):
        for h in (1, 2, 6):
            add("v_hist", f"h{h}", f"hist = {h}", "spatial", hist=h)
        add("v_hist", "control", f"hist = {HIST_DEFAULT} (default)", "spatial", control=True)
        for a in (0.25, 0.75):
            add("v_amp", f"a{a}", f"season_amp = {a}", "spatial", season_amp=a)
        add("v_amp", "control", "season_amp = 0.5 (default)", "spatial", control=True)
        for lam in (0.0, 5e-4, 5e-2):
            add("v_lam", f"l{lam}", f"lambda_L = {lam:g}", "latent", lambda_L=lam)
        add("v_lam", "control", f"lambda_L = {LAMBDA_L:g} (default)", "latent", control=True)

    return cfgs


# ─────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────
def build_windows(C, k):
    """(T, k, N): months t-k+1 .. t (oldest first), front-padded. Causal."""
    idx = (torch.arange(C.shape[0])[:, None] + torch.arange(-k + 1, 1)[None, :]).clamp(min=0)
    return C[idx]


def load_data(dataset):
    cfg = DATA[dataset]
    for p in cfg.values():
        if not os.path.exists(p):
            sys.exit(f"Missing file: {p}\nRun this script from the project root.")
    crime = pd.read_csv(cfg["crime"])
    arrest = pd.read_csv(cfg["arrest"])
    regions = pd.read_csv(cfg["order"]).iloc[:, 0].astype(int).tolist()
    crime["region_id"] = crime["region_id"].astype(int)
    arrest["region_id"] = arrest["region_id"].astype(int)
    months = sorted(crime["month"].unique())
    N, T = len(regions), len(months)

    try:
        moy = [pd.Period(str(m), freq="M").month - 1 for m in months]
    except Exception:
        print("WARNING: could not parse month labels; assuming the series starts in January")
        moy = [i % 12 for i in range(T)]

    def build_matrix(df, col):
        pv = df.pivot_table(index="month", columns="region_id", values=col, aggfunc="first")
        pv = pv.reindex(index=months, columns=regions)
        return pv.fillna(0).values.astype(np.float32)

    C_all = torch.tensor(build_matrix(crime, "C"))
    L_all = torch.tensor(build_matrix(arrest, "L"))
    n_train = int(T * TRAIN_RATIO)
    n_val = int(T * VAL_RATIO)
    split_of = {i: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
                for i in range(T)}
    return dict(
        dataset=dataset, regions=regions, months=months, N=N, T=T, moy=moy,
        C_all=C_all, L_all=L_all,
        A_sp=torch.tensor(np.load(cfg["adj_sp"]), dtype=torch.float32),
        A_fc=torch.tensor(np.load(cfg["adj_fc"]), dtype=torch.float32),
        n_train=n_train, n_val=n_val, split_of=split_of)


# ─────────────────────────────────────────────────────────────────────
# TRAIN + EVALUATE ONE (config, seed)
# ─────────────────────────────────────────────────────────────────────
def metrics(y, p):
    return dict(
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        R2=float(r2_score(y, p)),
        sMAPE=float(np.mean(2 * np.abs(p - y) / (np.abs(y) + np.abs(p) + 1e-8)) * 100))


def run_one(cfg, seed, D, epochs, patience):
    torch.manual_seed(seed)
    np.random.seed(seed)
    T, n_train, n_val = D["T"], D["n_train"], D["n_val"]
    C_all, L_all = D["C_all"], D["L_all"]

    A = D["A_fc"] if cfg["adj"] == "fc" else D["A_sp"]
    if cfg["shuffle"]:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(A.shape[0], generator=g)
        A = A[perm][:, perm]

    kw = dict(hist=HIST_DEFAULT, seasonal=True)
    kw.update(cfg["mk"])
    model = CLS[cfg["variant"]](A, T=T, alpha=0.3, beta=0.6, **kw)
    model.set_calendar(D["moy"], n_train=n_train)
    Cwin = build_windows(C_all, kw["hist"])

    is_l = lambda n: n.startswith("L_net")
    groups = [{"params": [p for n, p in model.named_parameters() if not is_l(n)], "lr": LR}]
    lnet = [p for n, p in model.named_parameters() if is_l(n)]
    if lnet:
        groups.append({"params": lnet, "lr": LR_L})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    t_span = torch.tensor([0.0, 1.0])
    lam = cfg["lambda_L"]
    use_pen = (cfg["variant"] == "latent" and getattr(model, "use_latent", True) and lam > 0)

    def predict(gi):
        model.set_context(L_all[gi], t_idx=gi)
        C0 = model.init_state(Cwin[gi])
        return odeint(model, C0, t_span, method="rk4",
                      options={"step_size": ODE_STEP})[-1]

    def pair_loss(gi):
        return loss_fn(predict(gi), C_all[gi + 1])

    tr_idx = list(range(0, n_train - 1))
    va_idx = list(range(n_train, n_train + n_val - 1))

    best_val, best_state, bad, ep_run = float("inf"), None, 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        ep_run = epoch
        model.train()
        order = np.random.permutation(len(tr_idx))
        idxs = [tr_idx[p] for p in order]
        for s in range(0, len(idxs), BATCH_PAIRS):
            batch = idxs[s:s + BATCH_PAIRS]
            opt.zero_grad()
            loss = sum(pair_loss(g) for g in batch) / len(batch)
            if use_pen:
                loss = loss + lam * model.smoothness_loss(n_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(sum(pair_loss(g) for g in va_idx).item() / max(len(va_idx), 1))
        sched.step(val)
        if val < best_val:
            best_val, bad = val, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    rows, yt, yp = [], [], []
    with torch.no_grad():
        for i in range(T - 1):
            Cp = predict(i).numpy()
            sp = D["split_of"][i + 1]
            if sp == "test":
                yt.append(C_all[i + 1].numpy())
                yp.append(Cp)
            for region, v in zip(D["regions"], Cp):
                rows.append({"month": D["months"][i + 1], "region_id": region,
                             "C_gnode": float(v), "split": sp})
    pred_path = (f"data/processed/{D['dataset']}_ablation_gnode_"
                 f"{cfg['name']}_seed{seed}_predictions.csv")
    pd.DataFrame(rows).to_csv(pred_path, index=False)

    m = metrics(np.concatenate(yt).ravel(), np.concatenate(yp).ravel())
    return {"dataset": D["dataset"], "name": cfg["name"], "block": cfg["block"],
            "label": cfg["label"], "role": cfg["role"], "variant": cfg["variant"],
            "seed": seed, **{k: round(v, 5) for k, v in m.items()},
            "best_val_mse": round(best_val, 6), "epochs_run": ep_run,
            "train_time_sec": round(train_time, 1)}


# ─────────────────────────────────────────────────────────────────────
# SUMMARY / TABLE
# ─────────────────────────────────────────────────────────────────────
def upsert(new_rows, path):
    new = pd.DataFrame(new_rows)
    if Path(path).exists():
        old = pd.read_csv(path)
        key = set(zip(new.dataset, new.name, new.seed))
        keep = [(d, n, s) not in key for d, n, s in zip(old.dataset, old.name, old.seed)]
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)


def build_table(summ, dataset):
    s = summ[summ.dataset == dataset].copy()
    if s.empty:
        return s
    ctrl = (s[s.role == "control"][["block", "seed", "R2"]]
            .rename(columns={"R2": "R2_ctrl"}))
    s = s.merge(ctrl, on=["block", "seed"], how="left")
    s["dR2"] = np.where(s.role == "control", np.nan, s.R2 - s.R2_ctrl)
    order = {n: i for i, n in enumerate(c["name"] for c in build_configs("all"))}
    tab = (s.groupby("name", sort=False)
             .agg(block=("block", "first"), label=("label", "first"), role=("role", "first"),
                  n_seeds=("seed", "nunique"),
                  MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
                  RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                  R2=("R2", "mean"), R2_sd=("R2", "std"),
                  sMAPE=("sMAPE", "mean"), sMAPE_sd=("sMAPE", "std"),
                  dR2=("dR2", "mean"), dR2_sd=("dR2", "std"))
             .reset_index())
    tab["_b"] = tab.block.map(lambda b: BLOCK_ORDER.index(b) if b in BLOCK_ORDER else 99)
    tab["_o"] = tab.name.map(lambda n: order.get(n, 999))
    return tab.sort_values(["_b", "_o"]).drop(columns=["_b", "_o"]).reset_index(drop=True)


def print_table(tab, dataset):
    print(f"\n===== {dataset.upper()}  GN-ODE ablations (test, mean over seeds) =====")
    for b, sub in tab.groupby("block", sort=False):
        print(f"\n-- {BLOCK_TITLES.get(b, b)}")
        for _, r in sub.iterrows():
            if np.isnan(r["dR2"]):
                d = "      -"
            else:
                sd = "" if (r["n_seeds"] < 2 or np.isnan(r["dR2_sd"])) else f" +/-{r['dR2_sd']:.4f}"
                d = f"{r['dR2']:+.4f}{sd}"
            print(f"  {r['label']:<42} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  "
                  f"R2={r['R2']:.4f}  sMAPE={r['sMAPE']:.2f}%  dR2={d}")


def write_latex(tab, path):
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Ablation & MAE$\downarrow$ & RMSE$\downarrow$ & $R^2\uparrow$ & "
         r"sMAPE (\%)$\downarrow$ & $\Delta R^2$ \\", r"\midrule"]
    for b, sub in tab.groupby("block", sort=False):
        L.append(r"\multicolumn{6}{l}{\textit{" + BLOCK_TITLES.get(b, b).replace("_", r"\_") + r"}} \\")
        for _, r in sub.iterrows():
            d = "--" if np.isnan(r["dR2"]) else f"{r['dR2']:+.4f}"
            lab = r["label"].replace("_", r"\_").replace("%", r"\%")
            L.append(f"{lab} & {r['MAE']:.4f} & {r['RMSE']:.4f} & {r['R2']:.4f} & "
                     f"{r['sMAPE']:.2f} & {d} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATA))
    ap.add_argument("--group", default="all",
                    choices=["dynamics", "graph", "temporal", "latent", "sensitivity", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--force", action="store_true", help="re-run finished (config, seed) pairs")
    ap.add_argument("--table_only", action="store_true",
                    help="skip training; rebuild the table from the existing summary CSV")
    args = ap.parse_args()

    os.makedirs("data/processed", exist_ok=True)
    stem = f"data/processed/{args.dataset}_ablation_gnode"
    SUMMARY, TABLE, TEX = f"{stem}_summary.csv", f"{stem}_table.csv", f"{stem}_table.tex"

    if not args.table_only:
        D = load_data(args.dataset)
        print(f"Dataset={args.dataset}  Regions={D['N']}  Months={D['T']}  "
              f"train/val/test = {D['n_train']}/{D['n_val']}/{D['T'] - D['n_train'] - D['n_val']}")
        cfgs = build_configs(args.group)
        print(f"Group='{args.group}': {len(cfgs)} configs x {len(args.seeds)} seed(s), "
              f"epochs={args.epochs}, patience={args.patience}")

        done = set()
        if Path(SUMMARY).exists() and not args.force:
            old = pd.read_csv(SUMMARY)
            done = set(zip(old[old.dataset == args.dataset].name,
                           old[old.dataset == args.dataset].seed))

        t_start = time.time()
        for seed in args.seeds:
            for cfg in cfgs:
                if (cfg["name"], seed) in done:
                    print(f"SKIP {cfg['name']:<22} seed={seed} (already done)")
                    continue
                print(f"RUN  {cfg['name']:<22} seed={seed}  {cfg['label']}")
                row = run_one(cfg, seed, D, args.epochs, args.patience)
                upsert([row], SUMMARY)          # saved after every run
                print(f"     MAE={row['MAE']}  RMSE={row['RMSE']}  R2={row['R2']}  "
                      f"sMAPE={row['sMAPE']}  ({row['epochs_run']} epochs, "
                      f"{row['train_time_sec']}s)")
        print(f"\nTotal wall-clock: {time.time() - t_start:.1f}s")

    if not Path(SUMMARY).exists():
        sys.exit("No summary file yet. Run without --table_only first.")
    tab = build_table(pd.read_csv(SUMMARY), args.dataset)
    tab.to_csv(TABLE, index=False)
    write_latex(tab, TEX)
    print_table(tab, args.dataset)
    print(f"\nSaved: {SUMMARY}\nSaved: {TABLE}\nSaved: {TEX}")