# -*- coding: utf-8 -*-
"""R1.3 消融：(b) 参考布局 + 逐路径 β 优化；(c) 随机可行布局分布。

覆盖模型与 matched_comparison.py 完全相同（1 mm 栅格、同一条视线规则、同一 N(x,y)），
该模型已能逐位复现已发表 Table II 的 RMSE（1.099/1.664/2.249/0.616），故可用于新增消融。
适应度按正文 F = 0.648 f1^P + 0.230 f2 + 0.122 f3；固定布局时 f3 为常数，不参与比较。
"""
import numpy as np

ROI = np.arange(0, 301, 1, dtype=float) + 100.0
CUT = 300.0
X, Y = np.meshgrid(ROI, ROI)
PLATE = ~((X > CUT) & (Y > CUT))

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


def path_rds(T, R):
    """预计算每条有效路径的 RD 场（依赖节点，不依赖 β）。

    返回 [(rd, k)]，k 为该路径在 36 条（传感器外层、激励内层）枚举中的原始下标，
    与 Table I / BETA_OPT 的顺序一致。无效路径被跳过，但下标不重排。
    """
    out = []
    k = 0
    for s in R:
        for t in T:
            a, b = np.array(t, float), np.array(s, float)
            if cross(a, b):
                k += 1
                continue
            d = np.linalg.norm(b - a)
            rd = (np.hypot(X - a[0], Y - a[1]) + np.hypot(X - b[0], Y - b[1])) / d
            out.append((rd, k))
            k += 1
    return out


def evaluate_rds(rds, betas, w):  # w 不使用，占位保持签名一致
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    for rd, beta in zip(rds, betas):
        m = rd < beta
        N += m
        P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    v = N[PLATE]
    e = P[PLATE]
    return dict(paths=len(rds),
                blind=100 * (v == 0).mean(),
                weak=100 * (v == 1).mean(),
                eff=100 * (v >= 2).mean(),
                rmseP=float(np.sqrt(np.mean((e - 2.0) ** 2))),
                meanN=float(v.mean()))


def fitness(res):
    # f1^P：加权场 P 低于 1 的像素比例；f2：RMSE(P)/2.5
    return 0.648 * res["f1P"] + 0.230 * res["rmseP"] / 2.5


def evaluate_with_f1(rds, betas):
    scalar = np.isscalar(betas)
    N = np.zeros_like(X)
    P = np.zeros_like(X)
    for rd, k in rds:
        beta = betas if scalar else betas[k]
        m = rd < beta
        N += m
        P += np.where(m, (beta - rd) / (beta - 1.0), 0.0)
    v = N[PLATE]
    e = P[PLATE]
    return dict(paths=len(rds),
                blind=100 * (v == 0).mean(),
                weak=100 * (v == 1).mean(),
                eff=100 * (v >= 2).mean(),
                rmseP=float(np.sqrt(np.mean((e - 2.0) ** 2))),
                meanN=float(v.mean()),
                f1P=float((e < 1.0).mean()),
                N=N)


GRID = [1.001, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030]


def optimise_betas(rds, sweeps=4, start=1.02, n_genes=36):
    betas = np.full(n_genes, start)
    cur = evaluate_with_f1(rds, betas)
    curF = fitness(cur)
    for _ in range(sweeps):
        improved = False
        for k in [kk for _, kk in rds]:
            best_b, best_F = betas[k], curF
            for cand in GRID:
                if abs(cand - betas[k]) < 1e-9:
                    continue
                trial = betas.copy()
                trial[k] = cand
                res = evaluate_with_f1(rds, trial)
                F = fitness(res)
                if F < best_F - 1e-9:
                    best_b, best_F, best_res = cand, F, res
            if best_b != betas[k]:
                betas[k] = best_b
                curF = best_F
                cur = best_res
                improved = True
        if not improved:
            break
    return betas, cur, curF


print("=" * 72)
print("(b) reference layout with optimised per-path beta")
rds_b = path_rds(T_BASE, R_BASE)
print(f"    valid paths: {len(rds_b)}/36")
base_single = evaluate_with_f1(rds_b, 1.02)
print(f"    single beta=1.02 : eff {base_single['eff']:.2f}%  "
      f"blind {base_single['blind']:.2f}%  weak {base_single['weak']:.2f}%  "
      f"RMSE(P) {base_single['rmseP']:.3f}")
betas_b, res_b, F_b = optimise_betas(rds_b)
print(f"    per-path (opt.)  : eff {res_b['eff']:.2f}%  "
      f"blind {res_b['blind']:.2f}%  weak {res_b['weak']:.2f}%  "
      f"RMSE(P) {res_b['rmseP']:.3f}")
print(f"    beta range {betas_b.min():.3f}-{betas_b.max():.3f}")

print("=" * 72)
print("(a2) optimized layout, relevant rows")
rds_o = path_rds(T_OPT, R_OPT)
for b in (1.01, 1.02, 1.03):
    r = evaluate_with_f1(rds_o, b)
    print(f"    single beta={b}: eff {r['eff']:.2f}%  blind {r['blind']:.2f}%  "
          f"weak {r['weak']:.2f}%  RMSE(P) {r['rmseP']:.3f}")
r = evaluate_with_f1(rds_o, BETA_OPT)
print(f"    per-path (Table I): eff {r['eff']:.2f}%  blind {r['blind']:.2f}%  "
      f"weak {r['weak']:.2f}%  RMSE(P) {r['rmseP']:.3f}")

print("=" * 72)
print("(c) distribution of random feasible layouts (beta=1.02, 6T + 6S)")


def sample_layout(rng, n=6, tries=4000):
    for _ in range(tries):
        Ts = []
        for _ in range(n):
            if rng.random() < 0.5:
                Ts.append((rng.uniform(10, 290), 450.0))
            else:
                Ts.append((rng.uniform(10, 490), 50.0))
        ok = all(np.hypot(Ts[i][0] - Ts[j][0], Ts[i][1] - Ts[j][1]) >= 50
                 for i in range(n) for j in range(i + 1, n))
        if not ok:
            continue
        Ss = []
        for _ in range(n):
            for _ in range(200):
                x, y = rng.uniform(0, 500), rng.uniform(0, 500)
                if x > CUT and y > CUT:
                    continue
                if abs(y - 50) < 5 or abs(y - 450) < 5:
                    continue
                if any(np.hypot(x - s[0], y - s[1]) < 10 for s in Ss):
                    continue
                if any(np.hypot(x - t[0], y - t[1]) < 10 for t in Ts):
                    continue
                Ss.append((x, y))
                break
            else:
                break
        if len(Ss) == n:
            return Ts, Ss
    return None


rng = np.random.default_rng(20261122)
effs, blinds, rmsp = [], [], []
n_ok = 0
while n_ok < 500:
    s = sample_layout(rng)
    if s is None:
        break
    Ts, Ss = s
    rds = path_rds(Ts, Ss)
    if len(rds) < 10:
        continue
    r = evaluate_with_f1(rds, 1.02)
    effs.append(r["eff"])
    blinds.append(r["blind"])
    rmsp.append(r["rmseP"])
    n_ok += 1

effs = np.array(effs)
blind = np.array(blinds)
print(f"    n = {n_ok} layouts")
print(f"    effective coverage: mean {effs.mean():.2f}  sd {effs.std(ddof=1):.2f}  "
      f"min {effs.min():.2f}  p5 {np.percentile(effs,5):.2f}  "
      f"median {np.median(effs):.2f}  p95 {np.percentile(effs,95):.2f}  max {effs.max():.2f}")
print(f"    blind zone: mean {blind.mean():.2f}  median {np.median(blind):.2f}")
print(f"    fraction beating 97.06%: {100*(effs>97.06).mean():.1f}%")
print(f"    fraction above reference 92.28%: {100*(effs>92.28).mean():.1f}%")
