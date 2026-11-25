"""Round-2 review verification (read-only).

Checks, against the published Table I and the MATLAB code in
F:\光致超声课题\论文\Joint Optimization ...\code\L型\ :
  1. which of the 36 t-r paths cross the cutout (x>300, y>300 mm),
     and by how much (overlap length, corner clearance);
  2. whether the 50-sample check in fun_ahd.m reproduces the 100-sample
     check in L_coverage_origin2.m;
  3. effective coverage recomputed with all 36 paths vs. LoS-valid paths only;
  4. the longest LoS-feasible path in the actual design space.
Unit: mm throughout.
"""
import numpy as np

# --- geometry -------------------------------------------------------------
PLATE = (0.0, 500.0, 0.0, 500.0)
CUT = (300.0, 500.0, 300.0, 500.0)      # removed: x>300, y>300
ROI = (100.0, 400.0, 100.0, 400.0)
CORNER = np.array([300.0, 300.0])

T = {  # transducers, mm (paper coordinates)
    "t1": (20, 450), "t2": (290, 450), "t3": (200, 450),
    "t4": (250, 50), "t5": (10, 50), "t6": (440, 50),
}
R = {
    "r1": (10, 280), "r2": (410, 130), "r3": (50, 390),
    "r4": (190, 480), "r5": (90, 40), "r6": (420, 240),
}

# Table I (paper) beta values, row = sensor, col = transducer t1..t6
BETA_TABLE = {
    "r1": [1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098],
    "r2": [1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213],
    "r3": [1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140],
    "r4": [1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0299],
    "r5": [1.0299, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057],
    "r6": [1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298],
}
# thresholds as hard-coded in L_coverage_origin2.m / L_stability_Origin3.m,
# flattened in the same order (sensor outer loop, transducer inner loop)
BETA_CODE = [
    1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,   # r1
    1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,   # r2
    1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,   # r3
    1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,   # r4
    1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,   # r5
    1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298,   # r6
]


def cut_overlap(a, b):
    """Length (mm) of the part of segment a-b strictly inside the cutout."""
    d = b - a
    lo, hi = 0.0, 1.0
    for p, q in ((-d[0], a[0] - CUT[0]), (d[0], CUT[1] - a[0]),
                 (-d[1], a[1] - CUT[2]), (d[1], CUT[3] - a[1])):
        if abs(p) < 1e-12:
            if q <= 0:
                return 0.0                     # parallel & outside
        else:
            t = q / p
            if p < 0:
                lo = max(lo, t)
            else:
                hi = min(hi, t)
    return max(0.0, (hi - lo)) * np.linalg.norm(d)


def corner_clearance(a, b):
    """Signed perpendicular distance of the segment from the corner (30,30)."""
    d = b - a
    t = np.clip(np.dot(CORNER - a, d) / np.dot(d, d), 0.0, 1.0)
    return np.linalg.norm(a + t * d - CORNER)


def sampled_in_cutout(a, b, n):
    """Reproduction of the 50-sample LoS test used in fun_ahd.m."""
    t = np.linspace(0, 1, n)
    pts = a + np.outer(t, b - a)
    inside = (pts[:, 0] > CUT[0]) & (pts[:, 1] > CUT[2])
    return bool(inside.any())


print("=" * 78)
print("1/2. line-of-sight status of the 36 paths")
print("=" * 78)
print(f"{'path':8s} {'len/mm':>8s} {'overlap/mm':>11s} "
      f"{'corner/mm':>10s} {'n=50':>6s} {'n=100':>6s}")
invalid = []
for sk, s in R.items():
    for tk, t in T.items():
        a, b = np.array(t, float), np.array(s, float)
        L = np.linalg.norm(b - a)
        ov = cut_overlap(a, b)
        cc = corner_clearance(a, b)
        c50 = sampled_in_cutout(a, b, 50)
        c100 = sampled_in_cutout(a, b, 100)
        flag = "CROSS" if ov > 0.05 else ("graze" if cc < 5 else "")
        if flag:
            print(f"{tk}-{sk:3s} {L:8.1f} {ov:11.3f} {cc:10.2f} "
                  f"{str(c50):>6s} {str(c100):>6s}  {flag}")
        if ov > 0.05:
            invalid.append((tk, sk))

print(f"\npaths with overlap > 0.05 mm : {len(invalid)}")
print("  " + ", ".join(f"{t}-{s}" for t, s in invalid))

print("\n" + "=" * 78)
print("3. effective coverage, ROI 300x300 mm, 1 mm grid")
print("=" * 78)


def coverage(paths, betas):
    xs = np.arange(ROI[0], ROI[1] + 0.1, 1.0)
    ys = np.arange(ROI[2], ROI[3] + 0.1, 1.0)
    X, Y = np.meshgrid(xs, ys)
    plate = ~((X > CUT[0]) & (Y > CUT[2]))   # cutout excluded from the statistics
    N = np.zeros_like(X)
    for (a, b), beta in zip(paths, betas):
        d = np.linalg.norm(b - a)
        rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
        N += (rd < beta).astype(float)
    N = N[plate]
    return (100 * np.mean(N == 0), 100 * np.mean(N == 1), 100 * np.mean(N >= 2))


all_paths, all_betas, ok_paths, ok_betas = [], [], [], []
k = 0
for sk, s in R.items():
    for tk, t in T.items():
        a, b = np.array(t, float), np.array(s, float)
        all_paths.append((a, b)); all_betas.append(BETA_CODE[k])
        if cut_overlap(a, b) <= 0.05:
            ok_paths.append((a, b)); ok_betas.append(BETA_CODE[k])
        k += 1

for label, p, bt in (("all 36 paths", all_paths, all_betas),
                     ("LoS-valid only", ok_paths, ok_betas)):
    z, w, e = coverage(p, bt)
    print(f"{label:15s} n={len(p):2d}  blind {z:5.2f}%  weak {w:5.2f}%  effective {e:5.2f}%")

print("\n" + "=" * 78)
print("4. longest LoS-feasible transducer-sensor path in the design space")
print("=" * 78)
cands_t, cands_s = [], []
for y in (50.0, 450.0):
    for x in np.arange(10.0, 491.0, 10.0):
        if x >= 300 and y >= 300:
            continue
        cands_t.append(np.array([x, y]))
cands_t = np.array(cands_t)
for y in np.arange(10.0, 491.0, 10.0):
    if y in (50.0, 450.0):
        continue
    for x in np.arange(10.0, 491.0, 10.0):
        if x >= 300 and y >= 300:
            continue
        cands_s.append(np.array([x, y]))
cands_s = np.array(cands_s)

best = (0.0, None)
for a in cands_t:
    d = np.linalg.norm(cands_s - a, axis=1)
    for idx in np.argsort(d)[::-1][:15]:
        b = cands_s[idx]
        if cut_overlap(a, b) <= 0.05:
            if d[idx] > best[0]:
                best = (d[idx], (tuple(a), tuple(b)))
            break
print(f"d_max = {best[0]:.1f} mm   from {best[1][0]} to {best[1][1]}")
beta_cap = 1.03
print(f"beta<{beta_cap} gives h = (d/2)sqrt(beta^2-1) = "
      f"{best[0] / 2 * np.sqrt(beta_cap ** 2 - 1):.1f} mm at d_max")
d_ref = 532.0
print(f"beta at h=80 mm for d=532 mm: {np.sqrt(1 + (2 * 80 / d_ref) ** 2):.4f}")
print(f"beta at h=80 mm for d={best[0]:.0f} mm: "
      f"{np.sqrt(1 + (2 * 80 / best[0]) ** 2):.4f}")
print(f"half-width at beta=1.001 for the 532 mm path: "
      f"{532 / 2 * np.sqrt(1.001 ** 2 - 1):.2f} mm "
      f"(S0 wavelength at 600 kHz, 5400 m/s = {5400e3 / 600e3:.1f} mm)")
