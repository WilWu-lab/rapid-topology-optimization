"""Export the clearance-constrained optimum in the format of data\\fig8_data.

Usage: python -I export_new_layout.py ga_L_seed11.npz new_layout
Writes <outdir>: x_mm.csv, y_mm.csv, P.csv, mask_*.csv, long.csv, summary.csv,
nodes.csv, betas.csv, matched.csv
"""
import os
import sys

import numpy as np

src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)

T_CAND = ([ (float(x), 50.0) for x in range(10, 491, 10) if x < 300 ] +
          [ (float(x), 450.0) for x in range(10, 291, 10) ])
S_CAND = [(float(x), float(y)) for x in range(10, 491, 10)
          for y in range(10, 491, 10)
          if not (x >= 300 and y >= 300) and y not in (50.0, 450.0)]
T_CAND = np.array(T_CAND)
S_CAND = np.array(S_CAND)
CUT, CORNER, CLR = 300.0, np.array([300.0, 300.0]), 5.0

T_BASE = [(100, 450), (200, 450), (100, 50), (200, 50), (300, 50), (400, 50)]
R_BASE = [(50, 400), (150, 400), (250, 400), (100, 100), (250, 100), (400, 100)]

GRID = np.arange(100.0, 400.0 + 1e-9, 1.0)
X, Y = np.meshgrid(GRID, GRID)
PLATE = ~((X > CUT) & (Y > CUT))


def geom(a, b):
    d = b - a
    lo, hi = 0.0, 1.0
    outside = False
    for p, q in ((-d[0], a[0] - CUT), (d[0], 500.0 - a[0]),
                 (-d[1], a[1] - CUT), (d[1], 500.0 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                outside = True
                break
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
    t = np.clip(np.dot(CORNER - a, d) / np.dot(d, d), 0.0, 1.0)
    clr = float(np.linalg.norm(a + t * d - CORNER))
    return (False if outside else (hi - lo) > 1e-6), clr


def fields(T, R, betas):
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    valid = []
    k = -1
    for s in R:
        for t in T:
            k += 1
            a, b = np.array(t, float), np.array(s, float)
            blocked, c = geom(a, b)
            if blocked or c < CLR:
                valid.append(np.nan)
                continue
            valid.append(betas[k])
            d = np.hypot(*(b - a))
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
            m = rd < betas[k]
            N += m
            P += np.where(m, (betas[k] - rd) / (betas[k] - 1.0), 0.0)
    return N, P, valid


def stats(N, P):
    v = N[PLATE]
    return (100 * (v == 0).mean(), 100 * (v == 1).mean(), 100 * (v >= 2).mean(),
            float(np.sqrt(np.mean((P[PLATE] - 2.0) ** 2))), float(v.mean()),
            int((v >= 0).sum()))


d = np.load(src)
t, s, b = d["t"], d["s"], d["b"]
T = [tuple(map(float, T_CAND[j])) for j in t]
R = [tuple(map(float, S_CAND[j])) for j in s]

N, P, valid = fields(T, R, b)
z, w, e, rm, mn, npix = stats(N, P)
print(f"layout from {src}")
print("transducers:", [(int(x), int(y)) for x, y in T])
print("sensors    :", [(int(x), int(y)) for x, y in R])
print(f"valid pairs: {sum(np.isfinite(valid))}/36")
print(f"blind {z:.2f}%  weak {w:.2f}%  effective {e:.2f}%  RMSE {rm:.3f}  meanN {mn:.3f}")

# --- write field files in the fig8_data layout -----------------------------
np.savetxt(os.path.join(outdir, "x_mm.csv"), GRID, delimiter=",", fmt="%.1f")
np.savetxt(os.path.join(outdir, "y_mm.csv"), GRID, delimiter=",", fmt="%.1f")
Nv = np.where(PLATE, N, np.nan)
np.savetxt(os.path.join(outdir, "P.csv"), Nv, delimiter=",", fmt="%.4f")
for name, mask in (("blind", Nv == 0), ("weak", Nv == 1), ("effective", Nv >= 2)):
    np.savetxt(os.path.join(outdir, f"mask_{name}.csv"),
               np.where(mask, 1.0, np.nan), delimiter=",", fmt="%.1f")
with open(os.path.join(outdir, "long.csv"), "w", encoding="utf-8") as fh:
    fh.write("x_mm,y_mm,N,blind,weak,effective\n")
    for i in range(GRID.size):
        for j in range(GRID.size):
            if not PLATE[i, j]:
                continue
            fh.write(f"{GRID[j]:.1f},{GRID[i]:.1f},{int(N[i, j])},"
                     f"{int(N[i, j] == 0)},{int(N[i, j] == 1)},{int(N[i, j] >= 2)}\n")

with open(os.path.join(outdir, "nodes.csv"), "w", encoding="utf-8") as fh:
    fh.write("node_type,x_mm,y_mm\n")
    for x, y in T:
        fh.write(f"transducer,{x:.1f},{y:.1f}\n")
    for x, y in R:
        fh.write(f"sensor,{x:.1f},{y:.1f}\n")

with open(os.path.join(outdir, "betas.csv"), "w", encoding="utf-8") as fh:
    fh.write("sensor,transducer,beta\n")
    k = -1
    for ii, sr in enumerate(R):
        for jj, tr in enumerate(T):
            k += 1
            val = valid[k]
            fh.write(f"r{ii+1}({sr[0]:.0f},{sr[1]:.0f}),"
                     f"t{jj+1}({tr[0]:.0f},{tr[1]:.0f}),"
                     f"{'invalid' if not np.isfinite(val) else f'{val:.4f}'}\n")

with open(os.path.join(outdir, "summary.csv"), "w", encoding="utf-8") as fh:
    fh.write("case,beta,blind_pct,weak_pct,effective_pct,rmse_target2,mean_path_count\n")
    fh.write(f"optimized,per-path,{z:.4f},{w:.4f},{e:.4f},{rm:.4f},{mn:.4f}\n")
    for bb in (1.01, 1.02, 1.03):
        N2, P2, _ = fields(T, R, [bb] * 36)
        z2, w2, e2, rm2, mn2, _ = stats(N2, P2)
        fh.write(f"optimized,{bb},{z2:.4f},{w2:.4f},{e2:.4f},{rm2:.4f},{mn2:.4f}\n")

# --- matched comparison ----------------------------------------------------
rows = []
for label, TT, RR in (("reference", T_BASE, R_BASE), ("optimized", T, R)):
    for bb in (1.01, 1.02, 1.03):
        N2, P2, _ = fields(TT, RR, [bb] * 36)
        z2, w2, e2, rm2, mn2, _ = stats(N2, P2)
        rows.append((label, bb, z2, w2, e2, rm2, mn2))
rows.append(("optimized", "per-path", z, w, e, rm, mn))
with open(os.path.join(outdir, "matched.csv"), "w", encoding="utf-8") as fh:
    fh.write("layout,beta,blind_pct,weak_pct,effective_pct,rmse_target2,mean_path_count\n")
    for r in rows:
        fh.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f}\n")
print("\nmatched comparison:")
for r in rows:
    print(f"  {r[0]:10s} beta={str(r[1]):>8s}  blind {r[2]:5.2f}  weak {r[3]:5.2f}  "
          f"eff {r[4]:6.2f}  RMSE {r[5]:5.3f}  meanN {r[6]:5.3f}")