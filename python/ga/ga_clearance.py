"""Re-optimization of the node layout with a corner-clearance constraint.

Mirrors the published model: 6 transducers on the two fibres, 6 off-fibre
sensors, 36 per-path shape factors, AHP weights (0.648 / 0.230 / 0.122),
1 cm candidate grid, d1 = 50 mm, d2 = d3 = 10 mm, L_lim = 30 mm.

Two differences from the submitted run, both deliberate:
  * a sensing path must clear the re-entrant corner (300, 300) by
    --clearance mm, not merely avoid entering the cut-out;
  * the fitness field is evaluated on a 5 mm grid during the search and the
    final individuals are re-scored on the 1 mm grid.

Usage:
  python -I ga_clearance.py --clearance 0    --tag ctl   --seed 1
  python -I ga_clearance.py --clearance 5    --tag clr5  --seed 1
"""
import argparse
import math

import numpy as np

P_WEIGHTS = (0.648, 0.230, 0.122)
E_MAX = 2.5
L_LIM = 30.0
D1, D2, D3 = 50.0, 10.0, 10.0
CUT = 300.0
CORNER = np.array([300.0, 300.0])

ap = argparse.ArgumentParser()
ap.add_argument("--clearance", type=float, default=5.0)
ap.add_argument("--tag", default="run")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--pop", type=int, default=200)
ap.add_argument("--gen", type=int, default=250)
ap.add_argument("--step", type=float, default=5.0, help="grid step, mm")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

# --- candidate grids -------------------------------------------------------
T_CAND = ([ (float(x), 50.0) for x in range(10, 491, 10) if x < 300 ] +
          [ (float(x), 450.0) for x in range(10, 291, 10) ])
S_CAND = [(float(x), float(y)) for x in range(10, 491, 10)
          for y in range(10, 491, 10)
          if not (x >= 300 and y >= 300) and y not in (50.0, 450.0)]
T_CAND = np.array(T_CAND)
S_CAND = np.array(S_CAND)

GRID = np.arange(100.0, 400.0 + 1e-9, args.step)
X, Y = np.meshgrid(GRID, GRID)
PLATE = ~((X > CUT) & (Y > CUT))
N_PIX = int(PLATE.sum())


def geom(a, b):
    """Return (crosses_cutout, clearance_to_corner) for segment a-b.

    The clearance is always computed: an axis-parallel path that lies outside
    the cut-out strip still has a well-defined distance to the corner.
    """
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
    blocked = False if outside else (hi - lo) > 1e-6
    return blocked, clr


_rd_cache = {}
_vcache = {}


def valid_pair(ti, si):
    key = (ti, si)
    v = _vcache.get(key)
    if v is None:
        blocked, clr = geom(T_CAND[ti], S_CAND[si])
        v = (not blocked) and (clr >= args.clearance)
        _vcache[key] = v
    return v


def rd_field(ti, si):
    key = (ti, si)
    f = _rd_cache.get(key)
    if f is None:
        a, b = T_CAND[ti], S_CAND[si]
        d = np.hypot(*(b - a))
        f = ((np.hypot(X - a[0], Y - a[1]) +
              np.hypot(X - b[0], Y - b[1])) / d).astype(np.float32)
        if len(_rd_cache) > 8000:
            _rd_cache.clear()
        _rd_cache[key] = f
    return f


def pairs_of(t_idx, s_idx):
    """All transducer-sensor pairs, sensor-major (matches Table I ordering)."""
    return [(int(ti), int(si)) for si in s_idx for ti in t_idx]


def fitness(ind):
    t_idx, s_idx, betas = ind
    P = np.zeros((GRID.size, GRID.size), dtype=np.float32)
    lengths = []
    k = -1
    for ti, si in pairs_of(t_idx, s_idx):
        k += 1
        if not valid_pair(ti, si):
            continue
        a, b = T_CAND[ti], S_CAND[si]
        lengths.append(np.hypot(*(b - a)))
        rd = rd_field(ti, si)
        beta = betas[k]
        m = rd < beta
        if m.any():
            P[m] += (beta - rd[m]) / (beta - 1.0)
    v = P[PLATE]
    f1 = float(np.mean(v < 1.0))
    f2 = min(1.0, float(np.sqrt(np.mean((v - 2.0) ** 2))) / E_MAX)
    if len(lengths) > 1:
        L = np.array(lengths)
        d = np.abs(L[:, None] - L[None, :])
        pen = np.maximum(0.0, L_LIM - d).sum() / 2.0
        f3 = pen / (len(L) * (len(L) - 1) / 2.0 * L_LIM)
    else:
        f3 = 1.0
    return P_WEIGHTS[0] * f1 + P_WEIGHTS[1] * f2 + P_WEIGHTS[2] * f3, (f1, f2, f3)


# --- constraint repair -----------------------------------------------------
def sep_ok(t_idx, s_idx):
    T = T_CAND[t_idx]
    S = S_CAND[s_idx]
    if len(t_idx) > 1:
        dt = np.hypot(*(T[:, None, :] - T[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(dt, np.inf)
        if dt.min() < D1:
            return False
    ds = np.hypot(*(S[:, None, :] - S[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(ds, np.inf)
    if ds.min() < D2:
        return False
    dx = np.hypot(*(T[:, None, :] - S[None, :, :]).transpose(2, 0, 1))
    return dx.min() >= D3


def random_ind():
    while True:
        t = rng.integers(0, len(T_CAND), 6)
        s = rng.integers(0, len(S_CAND), 6)
        if sep_ok(t, s):
            b = rng.uniform(1.001, 1.0299, 36)
            return [t, s, b]


def score(ind):
    f, _ = fitness(ind)
    return f


def mutate(ind):
    t, s, b = ind[0].copy(), ind[1].copy(), ind[2].copy()
    r = rng.random()
    if r < 0.2:
        i = rng.integers(0, 36)
        b[i] = rng.uniform(1.001, 1.0299)
    elif r < 0.6:
        s[rng.integers(0, 6)] = rng.integers(0, len(S_CAND))
    else:
        t[rng.integers(0, 6)] = rng.integers(0, len(T_CAND))
    if not sep_ok(t, s):
        return ind
    return [t, s, b]


def crossover(p1, p2):
    cp = rng.integers(1, 12)
    t = np.concatenate([p1[0][:cp] if cp < 6 else p1[0],
                        p2[0][cp:] if cp < 6 else np.array([], dtype=int)])
    t = t[:6] if len(t) == 6 else p1[0].copy()
    cp = rng.integers(1, 7)
    s = np.concatenate([p1[1][:cp], p2[1][cp:]])
    w = rng.random(36)
    b = w * p1[2] + (1 - w) * p2[2]
    if not sep_ok(t, s):
        return p1
    return [t, s, b]


pop = [random_ind() for _ in range(args.pop)]
fit = np.array([score(i) for i in pop])
best_hist = []
for g in range(args.gen):
    order = np.argsort(fit)
    elites = [pop[i] for i in order[:max(1, int(0.2 * args.pop))]]
    children = []
    while len(children) < args.pop - len(elites):
        def pick():
            c = rng.integers(0, args.pop, 4)
            return pop[c[np.argmin(fit[c])]]
        ch = crossover(pick(), pick())
        if rng.random() < 0.15:
            ch = mutate(ch)
        children.append(ch)
    pop = elites + children
    fit = np.array([score(i) for i in pop])
    best_hist.append(fit.min())
    if (g + 1) % 25 == 0:
        print(f"  gen {g+1:4d}  best F {fit.min():.4f}", flush=True)

i = int(np.argmin(fit))
t, s, b = pop[i]
print(f"\n[{args.tag}] best F = {fit[i]:.4f}")
print("transducers:", [(float(T_CAND[j][0]), float(T_CAND[j][1])) for j in t])
print("sensors    :", [(float(S_CAND[j][0]), float(S_CAND[j][1])) for j in s])

# --- verification on the 1 mm grid ----------------------------------------
GRID1 = np.arange(100.0, 400.0 + 1e-9, 1.0)
X1, Y1 = np.meshgrid(GRID1, GRID1)
PLATE1 = ~((X1 > CUT) & (Y1 > CUT))
N1 = np.zeros_like(X1)
P1 = np.zeros_like(X1)
used = []
k = -1
for ti, si in pairs_of(t, s):
    k += 1
    a, b_ = T_CAND[ti], S_CAND[si]
    blocked, clr = geom(a, b_)
    if blocked or clr < args.clearance:
        continue
    used.append(f"t{ti}-s{si}")
    d = np.hypot(*(b_ - a))
    rd = (np.hypot(X1 - a[0], Y1 - a[1]) + np.hypot(X1 - b_[0], Y1 - b_[1])) / d
    beta = b[k]
    m = rd < beta
    N1 += m
    P1 += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
v = N1[PLATE1]
print(f"\n[1 mm grid] paths {len(used)}  blind {100*(v==0).mean():.2f}%  "
      f"weak {100*(v==1).mean():.2f}%  eff {100*(v>=2).mean():.2f}%  "
      f"RMSE {np.sqrt(np.mean((P1[PLATE1]-2)**2)):.3f}  meanN {v.mean():.3f}")
np.savez(f"ga_{args.tag}_seed{args.seed}.npz", t=t, s=s, b=b,
         hist=np.array(best_hist))