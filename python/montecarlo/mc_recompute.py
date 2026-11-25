"""Monte Carlo recomputation for Reviewer 2 Comments 5 and 6.

Rule reproduced from L_stability_Origin3.m:
  - transducer: Gaussian displacement along x only, sigma = 20/3 mm,
    truncated at |dx| <= 20 mm, resampling until the node stays on the plate;
  - sensor: isotropic Gaussian, sigma = 20/3 mm per axis, resampling until
    |(dx,dy)| <= 20 mm and the node stays on the plate;
  - every perturbed layout is re-checked for the cutout-crossing constraint
    (a path through x>300 & y>300 is invalid) and for the minimum separations
    d1 = 50 mm, d2 = d3 = 10 mm;
  - coverage is evaluated on the 1 mm ROI grid (100..400 mm), cutout excluded.

Run: python -I mc_recompute.py [n_runs]
"""
import sys
import numpy as np

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED = 20261119
SIGMA = 20.0 / 3.0
CAP = 20.0
D1, D2, D3 = 50.0, 10.0, 10.0
CUT = 300.0

T_OPT = np.array([(20, 450), (290, 450), (200, 450),
                  (250, 50), (10, 50), (440, 50)], float)
R_OPT = np.array([(10, 280), (410, 130), (50, 390),
                  (190, 480), (90, 40), (420, 240)], float)
T_BASE = np.array([(100, 450), (200, 450), (100, 50),
                   (200, 50), (300, 50), (400, 50)], float)
R_BASE = np.array([(50, 400), (150, 400), (250, 400),
                   (100, 100), (250, 100), (400, 100)], float)
BETA_TABLE = np.array([
    1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,
    1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,
    1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,
    1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,
    1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,
    1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298])

ROI = np.arange(100.0, 401.0, 1.0)
X, Y = np.meshgrid(ROI, ROI)
PLATE = ~((X > CUT) & (Y > CUT))
MASK = PLATE
N_PIX = int(MASK.sum())


def on_plate(p):
    return not (p[0] > CUT and p[1] > CUT) and 0 <= p[0] <= 500 and 0 <= p[1] <= 500


def perturb(rng, pts, axis):
    out = np.empty_like(pts)
    for i, p in enumerate(pts):
        while True:
            if axis == "x":                       # transducers: x only
                dx = rng.normal(0, SIGMA)
                dy = 0.0
                if abs(dx) > CAP:
                    continue
            else:                                 # sensors: isotropic
                dx, dy = rng.normal(0, SIGMA, 2)
                if np.hypot(dx, dy) > CAP:
                    continue
            q = p + np.array([dx, dy])
            if on_plate(q):
                out[i] = q
                break
    return out


def min_sep_ok(T, R):
    dt = np.hypot(*(T[:, None, :] - T[None, :, :]).transpose(2, 0, 1))
    dr = np.hypot(*(R[:, None, :] - R[None, :, :]).transpose(2, 0, 1))
    dx = np.hypot(*(T[:, None, :] - R[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(dt, np.inf)
    np.fill_diagonal(dr, np.inf)
    return dt.min() >= D1 and dr.min() >= D2 and dx.min() >= D3


def crosses(a, b):
    d = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-d[0], a[0] - CUT), (d[0], 500 - a[0]),
                 (-d[1], a[1] - CUT), (d[1], 500 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return False
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
    return hi - lo > 1e-9


def coverage(T, R, betas):
    N = np.zeros((301, 301))
    for k, (s, t) in enumerate([(s, t) for s in R for t in T]):
        if crosses(t, s):
            continue
        d = np.hypot(*(s - t))
        rd = (np.hypot(X - t[0], Y - t[1]) + np.hypot(X - s[0], Y - s[1])) / d
        N += (rd < betas[k]).astype(float)
    v = N[MASK]
    return 100.0 * (v == 0).mean(), 100.0 * (v == 1).mean(), 100.0 * (v >= 2).mean()


cases = [("optimized", T_OPT, R_OPT, BETA_TABLE),
         ("baseline (beta=1.02)", T_BASE, R_BASE, np.full(36, 1.02))]

rng = np.random.default_rng(SEED)
for name, T, R, betas in cases:
    res = np.empty((N_RUNS, 3))
    n_infeasible = 0
    for k in range(N_RUNS):
        while True:
            Tp = perturb(rng, T, "x")
            Rp = perturb(rng, R, "xy")
            if min_sep_ok(Tp, Rp):
                break
            n_infeasible += 1
        res[k] = coverage(Tp, Rp, betas)
        if (k + 1) % 200 == 0:
            print(f"  {name}: {k + 1}/{N_RUNS}", flush=True)
    names = ("blind", "weak", "effective")
    print(f"\n{name}  (n={N_RUNS}, seed={SEED}, "
          f"rejections={n_infeasible})")
    for j, nm in enumerate(names):
        print(f"  {nm:10s} mean {res[:, j].mean():6.2f}  "
              f"SD {res[:, j].std(ddof=1):5.2f}  "
              f"p5 {np.percentile(res[:, j], 5):6.2f}  "
              f"min {res[:, j].min():6.2f}")
    tag = "optimized" if name == "optimized" else "baseline_beta102"
    with open(f"mc_trials_{tag}.csv", "w", encoding="utf-8") as fh:
        fh.write("trial,blind_pct,weak_pct,effective_pct\n")
        for i, row in enumerate(res, 1):
            fh.write(f"{i},{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}\n")
    print(f"  -> mc_trials_{tag}.csv")