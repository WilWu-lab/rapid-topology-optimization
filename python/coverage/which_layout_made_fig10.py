"""Which node list actually produced the published Fig. 10 / Table III numbers?

L_RAPID_Location.m contains two node lists:
  (i)  the commented-out legacy block  t = [2,29,19,25,1,44], r = [1,42,5,19,9,42]
  (ii) the active block               t = [10,20,10,20,30,40], r = [10,20]  (collocated)
The coverage scripts use the GA output t3 = 20, r2 = 41.

We test every combination of t3 in {19,20} x r2 in {41,42} on the optimized
6-sensor layout with the per-path betas, and report which reproduces the
published (145,149) and (170,330).
"""
import numpy as np

BETA = np.array([
    1.0273, 1.0010, 1.0124, 1.0161, 1.0217, 1.0098,
    1.0037, 1.0010, 1.0299, 1.0011, 1.0142, 1.0213,
    1.0203, 1.0077, 1.0070, 1.0131, 1.0163, 1.0140,
    1.0272, 1.0253, 1.0142, 1.0258, 1.0048, 1.0300,
    1.0300, 1.0038, 1.0097, 1.0090, 1.0114, 1.0057,
    1.0128, 1.0162, 1.0097, 1.0011, 1.0179, 1.0298])

DI = {
    (175, 325): [0.0389, 0.0028, 0.0163, 0.0028, 0.0069, 0.0167, 0.1244, 0.0001,
                 0.0001, 0.0644, 0.0051, 0.0180, 0.0198, 0.0006, 0.0001, 0.0129,
                 0.0001, 0.0028, 0.0001, 0.0001, 0.0017, 0.0228, 0.0401, 0.0272,
                 0.0001, 0.0253, 0.1457, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
                 0.0001, 0.0235, 0.0120, 0.03],
    (150, 130): [0.0988, 0.2574, 0.0571, 0.2574, 0.0001, 0.0996, 0.0159, 0.0001,
                 0.0001, 0.0503, 0.0224, 0.0575, 0.0011, 0.0029, 0.0001, 0.0024,
                 0.0001, 0.0059, 0.0001, 0.0001, 0.0179, 0.0590, 0.0900, 0.0563,
                 0.0001, 0.1181, 0.0056, 0.0001, 0.0696, 0.0001, 0.0001, 0.0001,
                 0.0001, 0.0612, 0.0322, 0.0294],
}

x = np.arange(0, 50.0, 0.1)
y = np.arange(0, 50.0, 0.1)
X, Y = np.meshgrid(x, y)
inside = ~((X > 30.0) & (Y > 30.0))


def reconstruct(di, T, R, betas):
    P = np.zeros((500, 500))
    k = 0
    for s in R:
        for t in T:
            b = betas[k]
            k += 1
            a, c = np.array(t, float), np.array(s, float)
            blocked = ((np.linspace(a[0], c[0], 400) > 30.01) &
                       (np.linspace(a[1], c[1], 400) > 30.01)).any()
            if blocked:
                continue
            d = np.hypot(*(c - a))
            if d < 1e-6:
                continue
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - c[0], Y - c[1])) / d
            w = np.where(rd < b, (b - rd) / (b - 1.0), 0.0)
            P += di[k - 1] * w * inside
    P2 = P.copy()
    P2[:100, :] = 0
    P2[:, :100] = 0
    P2[401:, :] = 0
    P2[:, 401:] = 0
    P2 *= inside
    r, cc = np.unravel_index(np.argmax(P2), P2.shape)
    return x[cc] * 10, y[r] * 10


print("published:  D1 (150,130) -> (145,149), 19.7 mm | D2 (175,325) -> (170,330), 7.1 mm")
print()
print(f"{'t3 x':>5s} {'r2 x':>5s} {'D1 estimate':>16s} {'err':>6s} "
      f"{'D2 estimate':>16s} {'err':>6s}")
for t3 in (19.0, 20.0):
    for r2 in (41.0, 42.0):
        T = [(2.0, 45.0), (29.0, 45.0), (t3, 45.0), (25.0, 5.0), (1.0, 5.0), (44.0, 5.0)]
        R = [(1.0, 28.0), (r2, 13.0), (5.0, 39.0), (19.0, 48.0), (9.0, 4.0), (42.0, 24.0)]
        e1 = reconstruct(DI[(150, 130)], T, R, BETA)
        e2 = reconstruct(DI[(175, 325)], T, R, BETA)
        print(f"{t3:5.0f} {r2:5.0f} "
              f"({e1[0]:6.1f},{e1[1]:6.1f}) "
              f"{np.hypot(e1[0]-150, e1[1]-130):6.2f} "
              f"({e2[0]:6.1f},{e2[1]:6.1f}) "
              f"{np.hypot(e2[0]-175, e2[1]-325):6.2f}")