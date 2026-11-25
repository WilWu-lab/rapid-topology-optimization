"""Matched-setting coverage table for Reviewer 2 Comment 3.

Both layouts are evaluated on the same 1 mm ROI grid, the same line-of-sight
rule (a path crossing x>300 & y>300 is invalid) and the same path-count field
N(x,y). Only the beta assignment differs between rows.
"""
import numpy as np

ROI = np.arange(0, 301, 1, dtype=float) + 100.0   # 100..400 mm
CUT = 300.0
T_OPT = [(20, 450), (290, 450), (200, 450), (250, 50), (10, 50), (440, 50)]
R_OPT = [(10, 280), (410, 130), (50, 390), (190, 480), (90, 40), (420, 240)]
T_BASE = [(100, 450), (200, 450), (100, 50), (200, 50), (300, 50), (400, 50)]
R_BASE = [(50, 400), (150, 400), (250, 400), (100, 100), (250, 100), (400, 100)]
BETA_OPT = [1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,
            1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,
            1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,
            1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,
            1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,
            1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298]

X, Y = np.meshgrid(ROI, ROI)
PLATE = ~((X > CUT) & (Y > CUT))


def cross(a, b):
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


def pairs(T, R):
    out = []
    for s in R:
        for t in T:
            a, b = np.array(t, float), np.array(s, float)
            out.append((a, b, not cross(a, b)))
    return out


def evaluate(T, R, betas):
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    used = 0
    for k, (a, b, ok) in enumerate(pairs(T, R)):
        if not ok:
            continue
        used += 1
        beta = betas[k]
        d = np.linalg.norm(b - a)
        rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
        m = rd < beta
        N += m.astype(float)
        P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    N = np.where(PLATE, N, np.nan)
    P = np.where(PLATE, P, np.nan)
    v = N[PLATE]
    return dict(paths=used,
                blind=100 * (v == 0).mean(),
                weak=100 * (v == 1).mean(),
                eff=100 * (v >= 2).mean(),
                rmseN=float(np.sqrt(np.mean((v - 2.0) ** 2))),
                rmseP=float(np.sqrt(np.mean((P[PLATE] - 2.0) ** 2))),
                meanN=float(v.mean()),
                meanP=float(P[PLATE].mean()), N=N)


opt = pairs(T_OPT, R_OPT)
base = pairs(T_BASE, R_BASE)
print(f"valid pairs: optimized {sum(p[2] for p in opt)}/36, "
      f"baseline {sum(p[2] for p in base)}/36")
print(f"pixels in ROI on plate: {int(PLATE.sum())}\n")
print(f"{'setting':34s} {'paths':>5s} {'blind%':>7s} {'weak%':>7s} "
      f"{'eff%':>7s} {'RMSE_N':>6s} {'RMSE_P':>6s} {'meanN':>6s} {'meanP':>6s}")
rows = {}
for layout, T, R, pf in (("baseline", T_BASE, R_BASE, base),
                         ("optimized", T_OPT, R_OPT, opt)):
    for b in (1.01, 1.02, 1.03):
        res = evaluate(T, R, [b] * 36)
        rows[(layout, b)] = res
        print(f"{layout + ', single beta=' + str(b):34s} {res['paths']:5d} "
              f"{res['blind']:7.2f} {res['weak']:7.2f} {res['eff']:7.2f} "
              f"{res['rmseN']:6.3f} {res['rmseP']:6.3f} "
              f"{res['meanN']:6.3f} {res['meanP']:6.3f}")
res = evaluate(T_OPT, R_OPT, BETA_OPT)
rows[("optimized", "perpath")] = res
print(f"{'optimized, per-path beta (Table I)':34s} {res['paths']:5d} "
      f"{res['blind']:7.2f} {res['weak']:7.2f} {res['eff']:7.2f} "
      f"{res['rmseN']:6.3f} {res['rmseP']:6.3f} "
      f"{res['meanN']:6.3f} {res['meanP']:6.3f}")

print("\n--- effect decomposition (effective coverage, %) ---")
for b in (1.01, 1.02, 1.03):
    g = rows[("optimized", b)]["eff"]
    bl = rows[("baseline", b)]["eff"]
    print(f"beta={b}: geometry effect {g - bl:+6.2f} pp "
          f"(optimized {g:.2f} vs baseline {bl:.2f})")
print(f"beta=1.02 freedom effect "
      f"{rows[('optimized','perpath')]['eff'] - rows[('optimized',1.02)]['eff']:+6.2f} pp")