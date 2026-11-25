"""Verify a GA result: node list, minimum corner clearance, coverage at 1 mm."""
import sys

import numpy as np

T_CAND = ([ (float(x), 50.0) for x in range(10, 491, 10) if x < 300 ] +
          [ (float(x), 450.0) for x in range(10, 291, 10) ])
S_CAND = [(float(x), float(y)) for x in range(10, 491, 10)
          for y in range(10, 491, 10)
          if not (x >= 300 and y >= 300) and y not in (50.0, 450.0)]
T_CAND = np.array(T_CAND)
S_CAND = np.array(S_CAND)
CUT, CORNER = 300.0, np.array([300.0, 300.0])

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


def coverage(t, s, betas, clr):
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    n = 0
    k = -1
    for si in s:
        for ti in t:
            k += 1
            a, b = T_CAND[ti], S_CAND[si]
            blocked, c = geom(a, b)
            if blocked or c < clr:
                continue
            n += 1
            d = np.hypot(*(b - a))
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
            beta = betas[k]
            m = rd < beta
            N += m
            P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    v = N[PLATE]
    return (n, 100 * (v == 0).mean(), 100 * (v == 1).mean(),
            100 * (v >= 2).mean(), float(np.sqrt(np.mean((P[PLATE] - 2) ** 2))))


for tag in sys.argv[1:]:
    d = np.load(f"ga_{tag}.npz")
    t, s, b = d["t"], d["s"], d["b"]
    print(f"\n=== {tag} ===")
    print("transducers/mm:", [(float(T_CAND[j][0]), float(T_CAND[j][1])) for j in t])
    print("sensors/mm    :", [(float(S_CAND[j][0]), float(S_CAND[j][1])) for j in s])
    clrs = []
    for si in s:
        for ti in t:
            blocked, c = geom(T_CAND[ti], S_CAND[si])
            if not blocked:
                clrs.append(c)
    clrs = np.array(sorted(clrs))
    print(f"smallest clearances over valid paths: "
          f"{np.round(clrs[:6], 2).tolist()} mm")
    for clr in (5.0, 9.0, 15.0):
        n, z, w, e, rm = coverage(t, s, b, clr)
        print(f"  clearance >= {clr:4.1f} mm: paths {n:2d}  blind {z:5.2f}%  "
              f"weak {w:5.2f}%  eff {e:6.2f}%  RMSE {rm:5.3f}")