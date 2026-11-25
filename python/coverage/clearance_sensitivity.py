"""Coverage as a function of the minimum clearance required between a sensing
path and the re-entrant corner (300,300).

The submitted result keeps every path that does not enter the cut-out, including
the two that pass within a few millimetres of the corner (t3-r2, t6-r4). This
script shows how much the headline number depends on that rule.
"""
import numpy as np

ROI = np.arange(100.0, 401.0, 1.0)
X, Y = np.meshgrid(ROI, ROI)
PLATE = ~((X > 300.0) & (Y > 300.0))
CUT, CORNER = 300.0, np.array([300.0, 300.0])

T = [(20, 450), (290, 450), (200, 450), (250, 50), (10, 50), (440, 50)]
R410 = [(10, 280), (410, 130), (50, 390), (190, 480), (90, 40), (420, 240)]
R420 = [(10, 280), (420, 130), (50, 390), (190, 480), (90, 40), (420, 240)]
BETA = [1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,
        1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,
        1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,
        1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,
        1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,
        1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298]


def cut_overlap(a, b):
    d = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-d[0], a[0] - CUT), (d[0], 500 - a[0]),
                 (-d[1], a[1] - CUT), (d[1], 500 - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return 0.0
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
    return max(0.0, hi - lo) * np.linalg.norm(d)


def clearance(a, b):
    d = b - a
    t = np.clip(np.dot(CORNER - a, d) / np.dot(d, d), 0.0, 1.0)
    return float(np.linalg.norm(a + t * d - CORNER))


def evaluate(R, betas, clr, T=T):
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    used = []
    k = -1
    for s in R:
        for t in T:
            k += 1
            a, b = np.array(t, float), np.array(s, float)
            if cut_overlap(a, b) > 0.05 or clearance(a, b) < clr:
                continue
            used.append(f"t{T.index(t)+1}-r{R.index(s)+1}")
            beta = betas[k]
            d = np.linalg.norm(b - a)
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
            m = rd < beta
            N += m
            P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    v = N[PLATE]
    return (100 * (v == 0).mean(), 100 * (v == 1).mean(), 100 * (v >= 2).mean(),
            float(np.sqrt(np.mean((P[PLATE] - 2.0) ** 2))), len(used), used)


print(f"{'r2':>6s} {'clearance':>10s} {'paths':>6s} {'blind%':>7s} "
      f"{'weak%':>7s} {'eff%':>7s} {'RMSE':>6s}")
for tag, R in (("410", R410), ("420", R420)):
    for clr in (0.0, 1.5, 5.0, 9.0):
        z, w, e, rm, n, used = evaluate(R, BETA, clr)
        print(f"{tag:>6s} {clr:10.1f} {n:6d} {z:7.2f} {w:7.2f} {e:7.2f} {rm:6.3f}")
T_BASE = [(100, 450), (200, 450), (100, 50), (200, 50), (300, 50), (400, 50)]
R_BASE = [(50, 400), (150, 400), (250, 400), (100, 100), (250, 100), (400, 100)]
print("\nreference layout under the same clearance rule:")
for b in (1.01, 1.02, 1.03):
    for clr in (0.0, 1.5, 5.0):
        z, w, e, rm, n, used = evaluate(R_BASE, [b] * 36, clr, T=T_BASE)
        print(f"  beta={b}, clearance {clr:4.1f}: paths {n:2d}  blind {z:5.2f}%  "
              f"weak {w:5.2f}%  eff {e:5.2f}%  RMSE {rm:5.3f}")

print()
print("paths excluded at clearance 5 mm:")
for tag, R in (("410", R410), ("420", R420)):
    z, w, e, rm, n, used = evaluate(R, BETA, 5.0)
    dropped = [f"t{i+1}-r{j+1}" for j in range(6) for i in range(6)
               if f"t{i+1}-r{j+1}" not in used]
    print(f"  r2={tag}: {sorted(dropped)}")