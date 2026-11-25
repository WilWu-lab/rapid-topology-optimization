"""Monte Carlo robustness for a GA layout under the documented perturbation rule.

  transducers: Gaussian, sigma = 20/3 mm, truncated at 20 mm, along x only
  sensors    : Gaussian, sigma = 20/3 mm per axis, truncated at 20 mm
  a node pushed off the plate is redrawn; the minimum separations d1/d2/d3 are
  checked and a violating layout is redrawn
  every perturbed layout is re-checked for the cut-out and the 5 mm clearance

Usage: python -I mc_new_layout.py ga_L_seed12.npz 1000
"""
import sys

import numpy as np

src = sys.argv[1]
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
SEED = 20261119
SIG, CAP = 20.0 / 3.0, 20.0
CLR = 5.0
D1, D2, D3 = 50.0, 10.0, 10.0
CUT, CORNER = 300.0, np.array([300.0, 300.0])

T_CAND = ([ (float(x), 50.0) for x in range(10, 491, 10) if x < 300 ] +
          [ (float(x), 450.0) for x in range(10, 291, 10) ])
S_CAND = [(float(x), float(y)) for x in range(10, 491, 10)
          for y in range(10, 491, 10)
          if not (x >= 300 and y >= 300) and y not in (50.0, 450.0)]
T_CAND = np.array(T_CAND)
S_CAND = np.array(S_CAND)

d = np.load(src)
T = np.array([T_CAND[j] for j in d["t"]], float)
R = np.array([S_CAND[j] for j in d["s"]], float)
B = np.array(d["b"], float)

GRID = np.arange(100.0, 400.0 + 1e-9, 1.0)
X, Y = np.meshgrid(GRID, GRID)
PLATE = ~((X > CUT) & (Y > CUT))


def geom(a, b):
    dd = b - a
    lo, hi = 0.0, 1.0
    outside = False
    for p, q in ((-dd[0], a[0] - CUT), (dd[0], 500.0 - a[0]),
                 (-dd[1], a[1] - CUT), (dd[1], 500.0 - a[1])):
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
    t = np.clip(np.dot(CORNER - a, dd) / np.dot(dd, dd), 0.0, 1.0)
    clr = float(np.linalg.norm(a + t * dd - CORNER))
    return (False if outside else (hi - lo) > 1e-6), clr


def on_plate(p):
    return 0 <= p[0] <= 500 and 0 <= p[1] <= 500 and not (p[0] > CUT and p[1] > CUT)


def perturb(rng, pts, axis):
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        while True:
            if axis == "x":
                dx, dy = rng.normal(0, SIG), 0.0
                if abs(dx) > CAP:
                    continue
            else:
                dx, dy = rng.normal(0, SIG, 2)
                if np.hypot(dx, dy) > CAP:
                    continue
            q = p + np.array([dx, dy])
            if on_plate(q):
                out[i] = q
                break
    return out


def sep_ok(Tp, Rp):
    dt = np.hypot(*(Tp[:, None, :] - Tp[None, :, :]).transpose(2, 0, 1))
    ds = np.hypot(*(Rp[:, None, :] - Rp[None, :, :]).transpose(2, 0, 1))
    dx = np.hypot(*(Tp[:, None, :] - Rp[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(dt, np.inf)
    np.fill_diagonal(ds, np.inf)
    return dt.min() >= D1 and ds.min() >= D2 and dx.min() >= D3


def coverage(Tp, Rp):
    N = np.zeros_like(X)
    k = -1
    for s in Rp:
        for t in Tp:
            k += 1
            blk, c = geom(t, s)
            if blk or c < CLR:
                continue
            dd = np.hypot(*(s - t))
            rd = (np.hypot(X - t[0], Y - t[1]) + np.hypot(X - s[0], Y - s[1])) / dd
            N += rd < B[k]
    v = N[PLATE]
    return 100 * (v == 0).mean(), 100 * (v == 1).mean(), 100 * (v >= 2).mean()


rng = np.random.default_rng(SEED)
res = np.empty((N_RUNS, 3))
rej = 0
for k in range(N_RUNS):
    while True:
        Tp, Rp = perturb(rng, T, "x"), perturb(rng, R, "xy")
        if sep_ok(Tp, Rp):
            break
        rej += 1
    res[k] = coverage(Tp, Rp)
    if (k + 1) % 200 == 0:
        print(f"  {k+1}/{N_RUNS}", flush=True)

print(f"\n{src}  n={N_RUNS} seed={SEED} redraws={rej}")
for j, nm in enumerate(("blind", "weak", "effective")):
    print(f"  {nm:10s} mean {res[:, j].mean():6.2f}  SD {res[:, j].std(ddof=1):5.2f}  "
          f"p5 {np.percentile(res[:, j], 5):6.2f}  min {res[:, j].min():6.2f}")
tag = src.replace("ga_", "").replace(".npz", "")
with open(f"mc_trials_{tag}.csv", "w", encoding="utf-8") as fh:
    fh.write("trial,blind_pct,weak_pct,effective_pct\n")
    for i, row in enumerate(res, 1):
        fh.write(f"{i},{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}\n")