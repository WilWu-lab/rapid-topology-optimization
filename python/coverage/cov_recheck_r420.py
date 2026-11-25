"""Coverage recheck after correcting the second sensor to r2 = (420, 130) mm.

Same grid, same line-of-sight rule and same beta values as the submitted
manuscript; only r2 moves by 10 mm in x, so the comparison isolates that change.
"""
import numpy as np

ROI = np.arange(100.0, 401.0, 1.0)
X, Y = np.meshgrid(ROI, ROI)
PLATE = ~((X > 300.0) & (Y > 300.0))
CUT = 300.0

T_OPT = [(20, 450), (290, 450), (200, 450), (250, 50), (10, 50), (440, 50)]
R_OLD = [(10, 280), (410, 130), (50, 390), (190, 480), (90, 40), (420, 240)]
R_NEW = [(10, 280), (420, 130), (50, 390), (190, 480), (90, 40), (420, 240)]
T_BASE = [(100, 450), (200, 450), (100, 50), (200, 50), (300, 50), (400, 50)]
R_BASE = [(50, 400), (150, 400), (250, 400), (100, 100), (250, 100), (400, 100)]
BETA = [1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,
        1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,
        1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,
        1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,
        1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,
        1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298]


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


def evaluate(T, R, betas, tag):
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    invalid = []
    k = -1
    for s in R:
        for t in T:
            k += 1
            a, b = np.array(t, float), np.array(s, float)
            if crosses(a, b):
                invalid.append(f"t{T.index(t)+1}-r{R.index(s)+1}")
                continue
            beta = betas[k]
            d = np.linalg.norm(b - a)
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
            m = rd < beta
            N += m.astype(float)
            P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    v = N[PLATE]
    res = dict(tag=tag, paths=36 - len(invalid), blind=100 * (v == 0).mean(),
               weak=100 * (v == 1).mean(), eff=100 * (v >= 2).mean(),
               rmse=float(np.sqrt(np.mean((P[PLATE] - 2.0) ** 2))),
               meanN=float(v.mean()), invalid=invalid)
    print(f"{tag:34s} paths {res['paths']:2d}  blind {res['blind']:5.2f}%  "
          f"weak {res['weak']:5.2f}%  eff {res['eff']:6.2f}%  "
          f"RMSE {res['rmse']:5.3f}  meanN {res['meanN']:5.3f}")
    return res


print("invalid pairs (t_i-r_j):")
for tag, R in (("r2 = 410 (submitted)", R_OLD), ("r2 = 420 (corrected)", R_NEW)):
    ev = evaluate(T_OPT, R, BETA, f"optimized, per-path, {tag}")
    print(f"   -> {ev['invalid']}")
print()
for tag, R in (("r2 = 410", R_OLD), ("r2 = 420", R_NEW)):
    for b in (1.01, 1.02, 1.03):
        evaluate(T_OPT, R, [b] * 36, f"optimized, beta={b}, {tag}")
    print()
for b in (1.01, 1.02, 1.03):
    evaluate(T_BASE, R_BASE, [b] * 36, f"reference, beta={b}")

print("\n--- sensitivity: r2 = 420 with the marginal path t3-r2 kept valid ---")
_orig_crosses = crosses
crosses = lambda a, b: False if (abs(a[0] - 200) < 1e-9 and abs(b[0] - 420) < 1e-9) else _orig_crosses(a, b)
evaluate(T_OPT, R_NEW, BETA, "optimized, per-path, r2=420, t3-r2 kept")
for b in (1.01, 1.02, 1.03):
    evaluate(T_OPT, R_NEW, [b] * 36, f"optimized, beta={b}, r2=420, t3-r2 kept")