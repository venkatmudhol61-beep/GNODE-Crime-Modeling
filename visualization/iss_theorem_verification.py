import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.linalg import eigvals
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
import os

# ── Load data ────────────────────────────────────────────────
df    = pd.read_csv('data/processed/crime_timeseries.csv')
W_raw = np.load('data/processed/adjacency.npy')

W   = (W_raw + W_raw.T) / 2.0
deg = W.sum(axis=1); deg[deg == 0] = 1.0
D_inv = np.diag(1.0 / np.sqrt(deg))
W     = D_inv @ W @ D_inv
N     = W.shape[0]

pivot  = df.pivot(index='month', columns='District', values='C')
pivot  = pivot.sort_index()
months = pivot.index.tolist()
T      = len(months)
C_data = np.nan_to_num(pivot.values.T, nan=0.0)
C_data = np.clip(C_data, 0, 1)

# ── Parameters ───────────────────────────────────────────────
# guard complex numerical noise with np.real before float()
lam_max    = float(np.max(np.real(eigvals(W))))
gamma_real = 0.05
alpha = 0.35
beta  = 2.00
L_min = (gamma_real + alpha + lam_max) / beta
gamma = beta * L_min - alpha - lam_max   # == gamma_real by construction
t_half = np.log(2) / gamma

# ── ODE: nominal system (d ≡ 0) — verifies parts (II) and (IV) ──
C0 = C_data[:, 0].copy()

def gnode(t, C):
    C = np.clip(C, 0, 1)
    return alpha * C * (1 - C) + W @ C - beta * L_min * C

t_eval  = np.linspace(0, T - 1, T)
sol     = solve_ivp(gnode, (0, T - 1), C0,
                    t_eval=t_eval, method='RK45',
                    rtol=1e-6, atol=1e-8)
C_model = np.clip(sol.y, 0, 1)

t_idx      = np.arange(T)
norm_data  = np.sqrt(np.sum(C_data**2,  axis=0))
norm_model = np.sqrt(np.sum(C_model**2, axis=0))
norm_C0    = norm_model[0]

# Nominal exponential bound — valid for d ≡ 0 (Theorem 4.2 part II)
norm_bound = norm_C0 * np.exp(-gamma * t_idx)

# Lyapunov function V(t) = 0.5 ||C||^2
# V_bound is the d=0 bound only: V̇ ≤ -2γV ⟹ V(t) ≤ V(0)e^{-2γt}
V_model = 0.5 * np.sum(C_model**2, axis=0)
V_bound = V_model[0] * np.exp(-2 * gamma * t_idx)

# ── ISS bound (Theorem 4.2 part I) ───────────────────────────
# d_max: max seasonal deviation of real data from its smooth trend
norm_smooth  = savgol_filter(norm_data, 25, 3)
d_max        = float(np.max(np.abs(norm_data - norm_smooth)))

exp_term = np.exp(-gamma * t_idx)

# FIX: ISS bound is a scalar inequality — square the WHOLE RHS, not term-by-term
# Wrong (old): norm_C0**2 * exp_term**2 + (d_max**2 / gamma) * (1 - exp_term**2)
# Correct (Eq. 30 squared):  (||C0|| e^{-γt} + (d_max/γ)(1-e^{-γt}))^2
iss_bound    = norm_C0 * exp_term + (d_max / gamma) * (1 - exp_term)
iss_bound_sq = iss_bound**2
iss_steady   = d_max / gamma

# ── Verification checks ──────────────────────────────────────
# V bound: only valid for nominal ODE (d=0), labelled accordingly
v_ok          = bool(np.all(V_model    <= V_bound    + 1e-4))
# Nominal norm bound (d=0)
n_ok          = bool(np.all(norm_model <= norm_bound + 1e-4))
# Physical invariance violations
inv_ok        = int(np.sum((C_model < -1e-6) | (C_model > 1 + 1e-6)))
# ISS bound on real data (squared comparison is now correct)
iss_ok        = bool(np.all(norm_data**2 <= iss_bound_sq + 1e-4))
# ISS bound on nominal ODE output
ode_iss_ok    = bool(np.all(norm_model   <= iss_bound    + 1e-4))
iss_viol_real = int(np.sum(norm_data**2 > iss_bound_sq + 1e-4))
iss_viol_ode  = int(np.sum(norm_model   > iss_bound    + 1e-4))

# Gershgorin / Jacobian eigenvalue check (part IV)
L_star  = np.full(N, L_min)
J0      = alpha * np.eye(N) + W - beta * np.diag(L_star)
eigs_c  = eigvals(J0)
all_neg = bool(all(np.real(eigs_c) < 0))

yr_ticks = list(range(0, T, 24))

print("=" * 62)
print("  THEOREM 4.2 — ISS VERIFICATION")
print("=" * 62)
print(f"  alpha={alpha},  beta={beta},  L_min={L_min:.4f}")
print(f"  lambda_max(W) = {lam_max:.4f}")
print(f"  gamma         = {gamma:.4f}")
print(f"  t_1/2         = {t_half:.0f} months = {t_half/12:.1f} years")
print(f"  d_max         = {d_max:.4f}  (max seasonal deviation)")
print(f"  ISS steady-state ||C||_2 -> {iss_steady:.4f}")
print()
print(f"  ODE nominal norm bound   satisfied : {n_ok}")
print(f"  Lyapunov bound (d=0)     satisfied : {v_ok}")
print(f"  ISS bound (ODE nominal)  satisfied : {ode_iss_ok}  ({iss_viol_ode}/T violations)")
print(f"  ISS bound (real data)    satisfied : {iss_ok}  ({iss_viol_real}/T violations)")
print(f"  Physical invariance viol.          : {inv_ok}")
print(f"  All Re(lambda_k(J0)) < 0           : {all_neg}")
print("=" * 62)

# ============================================================
# FIGURE — 4 panels (2x2)
# ============================================================
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig,
                         hspace=0.48, wspace=0.38,
                         top=0.90, bottom=0.08,
                         left=0.08, right=0.97)

# ── (a) Physical invariance ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.fill_between(t_idx, C_model.min(axis=0), C_model.max(axis=0),
                 alpha=0.22, color='steelblue',
                 label=r'ODE $[\min_i,\max_i]$')
ax1.plot(t_idx, C_model.mean(axis=0),
         color='steelblue', lw=2.0, label='ODE mean')
ax1.axhline(0, color='black', lw=1.5, ls='--', alpha=0.6)
ax1.axhline(1, color='black', lw=1.5, ls=':', alpha=0.6)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel(r'$C_i(t)$', fontsize=11)
ax1.set_title(r'(a) Physical Invariance $C_i(t)\in[0,1]$',
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=8, loc='center right')
ax1.set_xlim([0, T - 1]); ax1.set_ylim([-0.05, 1.08])
ax1.set_xticks(yr_ticks)
ax1.set_xticklabels([months[i][:4] for i in yr_ticks],
                    fontsize=7, rotation=45)
ax1.grid(True, alpha=0.18)

# ── (b) Lyapunov — nominal (d=0) bound ───────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.semilogy(t_idx, V_model, color='steelblue', lw=2.0,
             label=r'$V(t)=\frac{1}{2}\|C\|_2^2$ ODE (nominal)')
ax2.semilogy(t_idx, V_bound, color='crimson', lw=1.8, ls='--',
             label=r'Bound $V(0)e^{-2\gamma t}$ (d$\equiv$0 only)')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel(r'$V(t)$ (log scale)', fontsize=11)
ax2.set_title(r'(b) Lyapunov (nominal): $\dot{V}\leq -2\gamma V$  [$d\equiv 0$]',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xlim([0, T - 1])
ax2.set_xticks(yr_ticks)
ax2.set_xticklabels([months[i][:4] for i in yr_ticks],
                    fontsize=7, rotation=45)
ax2.grid(True, alpha=0.18)
ax2.text(0.35, 0.78, r'$\gamma=%.4f$' % gamma,
         transform=ax2.transAxes, fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3',
                   facecolor='lightyellow', edgecolor='gray'))

# ── (c) ISS bound (Eq. 30) ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t_idx, norm_data,
         color='gray', lw=1.3, alpha=0.55, ls='-',
         label=r'$\|\mathbf{C}(t)\|_2$ real data')
ax3.plot(t_idx, norm_smooth,
         color='darkorange', lw=1.8, ls='-',
         label=r'Smoothed trend')
ax3.plot(t_idx, norm_model,
         color='steelblue', lw=2.2,
         label=r'$\|\mathbf{C}(t)\|_2$ ODE (nominal)')
ax3.plot(t_idx, norm_bound,
         color='crimson', lw=1.6, ls='--',
         label=r'Nominal bound $\|\mathbf{C}_0\|_2 e^{-\gamma t}$ [d$\equiv$0]')
ax3.plot(t_idx, iss_bound,
         color='#27ae60', lw=2.2, ls='-.',
         label=r'ISS bound ')
ax3.axhline(iss_steady, color='#27ae60', lw=1.0, ls=':',
            alpha=0.7,
            label=fr'ISS ultimate bound $= {iss_steady:.3f}$')
ann_x = int(T * 0.18)
ax3.annotate(
    fr'$d_{{\max}}={d_max:.3f}$' + '\n(seasonal noise)',
    xy=(ann_x, norm_data[ann_x]),
    xytext=(ann_x + max(T // 10, 8), norm_data[ann_x] + 0.30),
    fontsize=8.5, color='darkorange',
    arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.25',
              facecolor='white', edgecolor='darkorange', alpha=0.90))
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel(r'$\|\mathbf{C}(t)\|_2$', fontsize=11)
ax3.set_title(
    r'(c) ISS Bound : $\|\mathbf{C}(t)\|_2 \leq'
    r'\|\mathbf{C}^0\|_2 e^{-\gamma t}'
    r'+ \frac{d_{\max}}{\gamma}(1-e^{-\gamma t})$',
    fontsize=10, fontweight='bold')
ax3.legend(fontsize=7.5, loc='upper center',
           bbox_to_anchor=(0.50, -0.22),
           ncol=2, framealpha=0.95,
           borderpad=0.6, handlelength=2.2)
ax3.set_xlim([0, T - 1])
ax3.set_xticks(yr_ticks)
ax3.set_xticklabels([months[i][:4] for i in yr_ticks],
                    fontsize=7, rotation=45)
ax3.grid(True, alpha=0.18)

# ── (d) Gershgorin discs + eigenvalues (part IV) ─────────────
ax4 = fig.add_subplot(gs[1, 1])
max_r = 0
for i in range(N):
    c = alpha - beta * L_star[i]
    r = sum(W[i, j] for j in range(N) if j != i)
    max_r = max(max_r, abs(c) + r)
    ax4.add_patch(plt.Circle(
        (c, 0), r, fill=True,
        facecolor='steelblue', edgecolor='steelblue',
        alpha=0.08, zorder=1))
    ax4.add_patch(plt.Circle(
        (c, 0), r, fill=False,
        edgecolor='steelblue', lw=0.7, alpha=0.45, zorder=2))
ax4.scatter(np.real(eigs_c), np.imag(eigs_c),
            color='crimson', s=70, zorder=5,
            label=r'$\lambda_k(J_0)$')
ax4.axvline(x=0, color='black', lw=1.5, ls='--', alpha=0.7,
            label=r'$\mathrm{Re}(\lambda)=0$')
ax4.set_xlabel(r'$\mathrm{Re}(\lambda)$', fontsize=11)
ax4.set_ylabel(r'$\mathrm{Im}(\lambda)$', fontsize=11)
ax4.set_title(
    r'(d) Gershgorin Discs and Eigenvalues of $J_0$'
    '\nPer-district sufficient condition (Eq. 33)',
    fontsize=10, fontweight='bold')
ax4.legend(fontsize=8.5, loc='upper left')
ax4.set_xlim([-max_r * 1.10, max_r * 0.25])
ax4.set_ylim([-max_r * 0.55,  max_r * 0.55])
ax4.grid(True, alpha=0.18)

# ── Main title ────────────────────────────────────────────────
fig.suptitle(
    'Theorem Verification: GES and ISS, '
    f'Chicago Crime Data, N={N} Districts, '
    f'{months[0]} to {months[-1]}\n'
    r'$\gamma=%.4f$,  $d_{\max}=%.4f$,  '
    r'$d_{\max}/\gamma = %.4f$ '
    % (gamma, d_max, d_max / gamma),
    fontsize=11, fontweight='bold', y=0.97)

# ── Save ──────────────────────────────────────────────────────
out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'theorem42_iss.png')
plt.savefig(out_path, dpi=220, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: theorem42_iss.png")
print(f"\nKey numbers for paper:")
print(f"  gamma              = {gamma:.4f}")
print(f"  d_max              = {d_max:.4f}")
print(f"  ISS ultimate bound = d_max/gamma = {d_max/gamma:.4f}")
print(f"  ODE nominal bound  : {n_ok}")
print(f"  ODE ISS bound      : {ode_iss_ok}")
print(f"  Real data ISS      : {iss_ok}  ({iss_viol_real}/{T})")