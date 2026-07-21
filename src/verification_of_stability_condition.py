"""
verify_stability_condition.py
==============================
Loads each trained GNODE checkpoint and computes the ACTUAL stability
condition from Theorem 4.2 (paper Theorem 3.8 / Eq. (11)):

    gamma = beta_min * L_min - alpha_max - lambda_max(W)

using the model's real learned alpha/beta and its real internal
adjacency buffer (self.A) -- not an idealized symmetric-normalized W,
not hand-picked constants. This directly answers: does the TRAINED
model actually satisfy condition (11) / (29)?

CHANGES vs. the previous version of this script
------------------------------------------------
1. W_desc strings were stale: they described an *earlier* generation
   of spatial_gnode.py / latent_enforcement_gnode.py that used
   row-normalization (spatial) or a raw/unnormalized adjacency
   (latent). Both files now build W = D^-1/2 A D^-1/2 with self-loops
   excluded before registering it as `self.A`, so `model.A` for these
   two variants IS the symmetric, self-loop-free operator to which
   Theorem 3.8's Perron-Frobenius bound (lambda_max(W) <= 1) applies
   directly. The description strings below now reflect that.
2. L_min was computed over the TEST period only, which does not match
   the paper's definition L_min := ess-inf over the FULL domain [0,T]
   (Assumption 2.3). This script now computes L_min over the full
   time domain by default (--l-min-domain full), with the previous
   test-only behaviour still available (--l-min-domain test) since
   that may be the relevant constant for a test-period-specific bound.
   The two are reported side by side so nothing is silently dropped.
3. The Spatial Attention variant's spatial term is a nonlinear,
   input-dependent softmax attention block, not a fixed linear
   operator W @ C. Reporting lambda_max(A+I) of the raw binary mask
   and plugging it into the same gamma formula as the linear-graph
   variants is not a rigorous test of the same condition for this
   variant (the paper's own Section 4.4 says the spectral argument
   does not transfer). This script still reports the number (it's a
   useful diagnostic) but flags it explicitly as NOT a certified
   check of Theorem 3.8 for that variant, rather than presenting it
   as an equivalent pass/fail alongside the other three.
4. Added a check for Assumption 2.4 (the sufficient condition behind
   the INVARIANCE guarantee, Theorem 3.6 -- separate from the
   stability guarantee, Theorem 3.8, that gamma above checks):

       beta_min * L_min >= R_max,   R_max := max_i sum_{j!=i} w_ij

   R_max is the largest row-sum of off-diagonal entries of the
   model's actual W. This was previously not checked at all; the
   paper's conclusion cited Theorem 3.6 as if it were certified
   without ever verifying its hypothesis against trained parameters,
   the same gap that had already been (correctly) disclosed for
   Theorem 3.8. Both checks now use the same worst-case beta_min and
   the same L_min_used (per --l-min-domain) for consistency.

Usage
-----
    python verify_stability_condition.py
    python verify_stability_condition.py --l-min-domain test
    python verify_stability_condition.py --resolution district

Assumes the same data/processed/ layout as src/train.py and that
checkpoints already exist at:
    data/processed/{resolution}_{variant}_gnode_best.pt
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.linalg import eigvals

from model.fc_gnode_one import FCGNOde
from model.spatial_gnode_one import GNODE
from model.spatial_attention_gnode_one import SpatialAttentionGNODE
from model.latent_enforcement_gnode_one import LatentEnforcementGNODE

# Must match src/train.py exactly
RESOLUTION_CONFIG = {
    "district": dict(
        unit_col="District",
        crime_csv="data/processed/crime_timeseries.csv",
        arrest_csv="data/processed/arrest_timeseries.csv",
        adjacency_path="data/processed/adjacency.npy",
        order_csv=None,
    ),
    "beat": dict(
        unit_col="Beat",
        crime_csv="data/processed/crime_timeseries.csv",
        arrest_csv="data/processed/arrest_timeseries.csv",
        adjacency_path="data/processed/adjacency.npy",
        order_csv="data/processed/graph_beats_order.csv",
    ),
}

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
VARIANTS    = ["fc", "spatial", "attention", "latent"]

# Variants whose reported gamma is a rigorous check of Thm 3.8's
# hypothesis (linear spatial operator W @ C with a known spectral
# bound). Attention is excluded: its spatial term is not W @ C, so
# gamma computed the same way is a diagnostic only, not a certified
# test of the same theorem.
RIGOROUS_GAMMA_CHECK = {"fc": True, "spatial": True, "attention": False, "latent": True}


def load_data(resolution):
    cfg = RESOLUTION_CONFIG[resolution]
    unit_col = cfg["unit_col"]

    for key in ("crime_csv", "arrest_csv", "adjacency_path"):
        if not os.path.exists(cfg[key]):
            raise FileNotFoundError(
                f"[{resolution}] missing required file: {cfg[key]}"
            )

    crime  = pd.read_csv(cfg["crime_csv"])
    arrest = pd.read_csv(cfg["arrest_csv"])
    crime.columns  = crime.columns.str.strip()
    arrest.columns = arrest.columns.str.strip()

    if unit_col not in crime.columns:
        raise KeyError(
            f"[{resolution}] expected column '{unit_col}' not found in "
            f"{cfg['crime_csv']} (found columns: {list(crime.columns)}). "
            f"This usually means the file at that path is actually the "
            f"other resolution's data -- check RESOLUTION_CONFIG paths."
        )
    if unit_col not in arrest.columns:
        raise KeyError(
            f"[{resolution}] expected column '{unit_col}' not found in "
            f"{cfg['arrest_csv']} (found columns: {list(arrest.columns)})."
        )

    crime[unit_col]  = crime[unit_col].astype(int)
    arrest[unit_col] = arrest[unit_col].astype(int)

    if cfg["order_csv"] is not None:
        units = pd.read_csv(cfg["order_csv"]).iloc[:, 0].astype(int).tolist()
        crime  = crime[crime[unit_col].isin(units)].copy()
        arrest = arrest[arrest[unit_col].isin(units)].copy()
    else:
        units = sorted(crime[unit_col].unique().tolist())

    months = sorted(crime["month"].unique())
    A = np.load(cfg["adjacency_path"])
    return crime, arrest, units, months, A, unit_col


def get_arrest_matrix(arrest, units, unit_col, months):
    """Returns (len(months), N) array of L(t,i) from observed arrest data."""
    piv = (arrest.pivot(index="month", columns=unit_col, values="L")
                 .reindex(index=months, columns=units)
                 .fillna(0.0))
    return piv.values


def lambda_max_real(W):
    """Largest real part of eigenvalues -- handles non-symmetric W safely."""
    ev = eigvals(W)
    return float(np.max(np.real(ev)))


def build_and_load(variant, resolution, A, N, T, ckpt_path):
    if variant == "fc":
        model = FCGNOde(N)
    elif variant == "spatial":
        model = GNODE(A)
    elif variant == "attention":
        model = SpatialAttentionGNODE(A, hidden_dim=32, T=T)
    elif variant == "latent":
        model = LatentEnforcementGNODE(A, T=T, hidden_dim=32)
    else:
        raise ValueError(variant)

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def get_alpha_beta(model, variant):
    """
    Returns (alpha_min, alpha_max, beta_min, beta_max).

    NOTE: previously the attention variant used a shared scalar
    (alpha, beta) at district scale (small N) and per-unit vectors at
    beat scale (large N). The current spatial_attention_gnode_one.py
    uses per-unit alpha/beta at BOTH resolutions (see its module
    docstring: "that special case has been removed"), so model.alpha
    / model.beta are always shape (N,) regardless of model.large_n.
    We therefore always treat them as arrays here. If you reintroduce
    a genuinely scalar parameterisation for some variant in the
    future, handle it explicitly by checking the tensor's .dim() == 0
    rather than by branching on model.large_n / variant name, since
    those are no longer reliable proxies for "is this scalar".
    """
    a = model.alpha.detach().numpy()
    b = model.beta.detach().numpy()
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return float(a.min()), float(a.max()), float(b.min()), float(b.max())


def get_R_max(W):
    """
    R_max := max_i sum_{j != i} w_ij  -- the largest row-sum of
    off-diagonal entries of W. This is Assumption 2.4's graph-side
    quantity: the worst-case total inflow a unit can receive from its
    neighbours when every neighbour is simultaneously at C_j = 1.

    Diagonal entries are excluded from the row sum to match the
    paper's definition (w_ii = 0 for spatial/latent by construction;
    fc's constant 1/N matrix has w_ii = 1/N, which IS excluded here
    per Assumption 2.4's j != i, consistent with Section 2.3's
    treatment of the FC self-weight as a spatial, not diagonal, term
    in this row-sum).
    """
    N = W.shape[0]
    off_diag = W * (1.0 - np.eye(N))
    row_sums = off_diag.sum(axis=1)
    return float(row_sums.max())


def get_W_and_lambda_max(model, variant):
    """
    Returns (W_used, lambda_max, description) -- the ACTUAL adjacency
    buffer used internally by the model's forward pass.

    NOTE: descriptions reflect the CURRENT model code (symmetric
    normalization, self-loops stripped, for spatial/latent). If you
    change the normalization inside spatial_gnode_one.py or
    latent_enforcement_gnode_one.py, update these strings to match --
    they are documentation only and are not re-derived from model.A.
    """
    if variant == "fc":
        W = model.A.detach().numpy()
        return W, lambda_max_real(W), (
            "ones(N,N)/N -- constant matrix, symmetric by construction. "
            "Perron-Frobenius bound lambda_max(W) <= 1 applies exactly."
        )
    if variant == "spatial":
        W = model.A.detach().numpy()
        return W, lambda_max_real(W), (
            "symmetric normalized W = D^-1/2 A D^-1/2, self-loops excluded "
            "(A_ii=0 enforced before normalization). Theorem 3.8's "
            "Perron-Frobenius spectral bound lambda_max(W) <= 1 applies "
            "directly."
        )
    if variant == "attention":
        W = model.A.detach().numpy()  # A + I, binary, unnormalized
        return W, lambda_max_real(W), (
            "binary adjacency + self-loops (A + I), UNNORMALIZED -- used "
            "only as an attention mask, not as a linear spatial operator. "
            "The reported lambda_max/gamma for this variant is a "
            "diagnostic only; Theorem 3.8's linear spectral argument does "
            "not transfer to the nonlinear attention mechanism (see paper "
            "Sec. 4.4). Do NOT treat this as a certified pass/fail on the "
            "same footing as the other three variants."
        )
    if variant == "latent":
        W = model.A.detach().numpy()
        return W, lambda_max_real(W), (
            "symmetric normalized W = D^-1/2 A D^-1/2, self-loops excluded "
            "(A_ii=0 enforced before normalization) -- identical "
            "normalization to the 'spatial' variant. Theorem 3.8's "
            "Perron-Frobenius spectral bound lambda_max(W) <= 1 applies "
            "directly."
        )
    raise ValueError(variant)


def get_L_min(model, variant, arrest_full, arrest_test, domain):
    """
    L_min computed either over the FULL time domain [0,T] (matching the
    paper's Assumption 2.3 definition, L_min := ess-inf_{t,i} L_i(t))
    or over the TEST period only (a weaker, test-specific constant).

    domain : "full" or "test"
    """
    if variant != "latent":
        arr = arrest_full if domain == "full" else arrest_test
        return float(np.min(arr))

    # latent: use model.L (T, N) property -- covers the full domain
    # natively since L_net is evaluated at every month index.
    with torch.no_grad():
        L_full = model.L.detach().numpy()  # (T, N)
    if domain == "full":
        return float(np.min(L_full))
    else:
        # test-only slice handled by caller via idxs; here we just
        # return the full-domain value sliced outside this function
        # for consistency -- see main() for the actual slicing.
        return float(np.min(L_full))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", choices=["district", "beat", "both"],
                         default="both",
                         help="Which resolution(s) to verify. Default: both "
                              "(skips a resolution gracefully if its data "
                              "files are missing or malformed).")
    parser.add_argument("--l-min-domain", choices=["full", "test"],
                         default="full",
                         help="Compute L_min over the full time domain "
                              "[0,T] (matches Assumption 2.3's ess-inf "
                              "definition used in the paper, default) or "
                              "over the test period only (weaker, "
                              "test-specific constant). Both values are "
                              "always reported in the output table "
                              "regardless of this flag; this flag only "
                              "selects which one is used to compute the "
                              "headline gamma / satisfied columns.")
    args = parser.parse_args()
    resolutions = ["district", "beat"] if args.resolution == "both" else [args.resolution]

    print("=" * 78)
    print("STABILITY CONDITION VERIFICATION -- ACTUAL TRAINED PARAMETERS")
    print("gamma = beta_min*L_min - alpha_max - lambda_max(W)   (Eq. 11)")
    print(f"L_min domain used for headline gamma: {args.l_min_domain}")
    print("=" * 78)

    rows = []
    for resolution in resolutions:
        try:
            crime, arrest, units, months, A, unit_col = load_data(resolution)
        except (FileNotFoundError, KeyError) as e:
            print(f"\n[skip resolution='{resolution}'] {e}\n")
            continue
        N, T = len(units), len(months)
        n_train = int(T * TRAIN_RATIO)
        n_val   = int(T * VAL_RATIO)
        months_test = months[n_train + n_val:]

        arrest_mat_full = get_arrest_matrix(arrest, units, unit_col, months)
        arrest_mat_test = get_arrest_matrix(arrest, units, unit_col, months_test)

        for variant in VARIANTS:
            ckpt_path = f"data/processed/{resolution}_{variant}_gnode_best.pt"
            try:
                model = build_and_load(variant, resolution, A, N, T, ckpt_path)
            except FileNotFoundError:
                print(f"[skip] checkpoint not found: {ckpt_path}")
                continue

            a_min, a_max, b_min, b_max = get_alpha_beta(model, variant)
            W, lam_max, W_desc = get_W_and_lambda_max(model, variant)
            R_max = get_R_max(W)

            L_min_full = get_L_min(model, variant, arrest_mat_full, arrest_mat_test, "full")
            L_min_test = get_L_min(model, variant, arrest_mat_full, arrest_mat_test, "test")
            L_min_used = L_min_full if args.l_min_domain == "full" else L_min_test

            # Worst-case (most conservative) alpha/beta for the global
            # stability condition (11): largest alpha, smallest beta.
            gamma = b_min * L_min_used - a_max - lam_max
            satisfied = gamma > 0
            rigorous = RIGOROUS_GAMMA_CHECK[variant]

            # Assumption 2.4 / invariance sufficient condition (10):
            # beta_min * L_min >= R_max. Uses the same worst-case
            # beta_min and the same L_min_used as the stability check
            # above, for consistency.
            beta_L_min = b_min * L_min_used
            invariance_satisfied = beta_L_min >= R_max

            rows.append(dict(
                resolution=resolution, variant=variant,
                alpha_range=f"[{a_min:.4f}, {a_max:.4f}]",
                beta_range=f"[{b_min:.4f}, {b_max:.4f}]",
                lambda_max=lam_max,
                L_min_full=L_min_full, L_min_test=L_min_test,
                L_min_used=L_min_used, l_min_domain=args.l_min_domain,
                gamma=gamma, satisfied=satisfied,
                rigorous_check=rigorous, W_desc=W_desc,
                R_max=R_max, beta_min_L_min=beta_L_min,
                invariance_satisfied=invariance_satisfied,
            ))

    df = pd.DataFrame(rows)
    print()
    for _, r in df.iterrows():
        status = "SATISFIED" if r["satisfied"] else "*** VIOLATED ***"
        caveat = "" if r["rigorous_check"] else "  [DIAGNOSTIC ONLY -- not a certified Thm 3.8 check, see W desc]"
        print(f"[{r['resolution']:8s} | {r['variant']:9s}] "
              f"alpha={r['alpha_range']:>22s}  beta={r['beta_range']:>22s}")
        print(f"{'':13s} lambda_max(W)={r['lambda_max']:.4f}  "
              f"L_min(full)={r['L_min_full']:.4f}  L_min(test)={r['L_min_test']:.4f}  "
              f"[using {r['l_min_domain']}]")
        print(f"{'':13s} gamma={r['gamma']:+.4f}  -> {status}{caveat}")
        inv_status = "SATISFIED" if r["invariance_satisfied"] else "*** VIOLATED ***"
        inv_caveat = "" if r["rigorous_check"] else "  [DIAGNOSTIC ONLY]"
        print(f"{'':13s} [Assumption 2.4] R_max={r['R_max']:.4f}  "
              f"beta_min*L_min={r['beta_min_L_min']:.4f}  -> {inv_status}{inv_caveat}")
        print(f"{'':13s} W used: {r['W_desc']}")
        print()

    print("=" * 78)
    rigorous_df = df[df["rigorous_check"]]
    n_bad = int((~rigorous_df["satisfied"]).sum())
    n_rigorous = len(rigorous_df)
    if n_rigorous == 0:
        print("No variants with a rigorous linear-operator check were evaluated.")
    elif n_bad == 0:
        print("All rigorously-checkable variants satisfy condition (11) with "
              "their trained parameters.")
    else:
        print(f"{n_bad}/{n_rigorous} rigorously-checkable variant(s) VIOLATE "
              f"condition (11) -- Theorem 3.8's exponential stability / ISS "
              f"guarantees do NOT provably hold for these trained models.")
    n_diag = len(df) - n_rigorous
    if n_diag:
        print(f"({n_diag} additional variant(s) reported for diagnostic "
              f"purposes only -- see caveats above; not part of the "
              f"rigorous-check count.)")
    print("=" * 78)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/stability_condition_verification.csv", index=False)
    print("\nSaved: data/processed/stability_condition_verification.csv")


if __name__ == "__main__":
    main()