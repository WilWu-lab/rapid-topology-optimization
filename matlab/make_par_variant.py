# -*- coding: utf-8 -*-
"""从 GA_r2_seed1.m 生成并行版 GA_r2_par.m（只改循环并行方式，算法语义不变）。

语义依据：fun_ahd 不含 rand/global/persistent（已核对），个体适应度彼此独立且确定，
因此把「种群评估」循环改成 parfor 不改变任何结果，只改变墙钟时间。
生成两处改动：
  1  SEED 由环境变量 GA_SEED 读入，供一个进程内连续跑多个种子
  2  两处种群评估 for → parfor，并在开头启动固定 worker 数的并行池
另外生成 3 代的冒烟测试副本 smoke/GA_smoke_par.m（仅用于测速，不进任何结果）。
"""
from pathlib import Path
import io
import sys

BASE = Path(r"F:\CherryInAcademic\projects\20261019_iotj_revision\r2\ga_seeds")
SRC = BASE / "GA_r2_seed1.m"
OUT = BASE / "GA_r2_par.m"
SMOKE_DIR = BASE / "smoke"

s = io.open(SRC, encoding="utf-8").read()

OLD_SEED = ("SEED = 1;                          % <<< 复制给别的种子时只改这一行 (2, 3)")
NEW_SEED = ("SEED = str2double(getenv('GA_SEED'));\n"
            "if isnan(SEED), SEED = 1; end\n"
            "NW = str2double(getenv('GA_WORKERS'));\n"
            "if isnan(NW) || NW < 1, NW = 20; end")
if s.count(OLD_SEED) != 1:
    print("[FAIL] seed line", s.count(OLD_SEED))
    sys.exit(1)
s = s.replace(OLD_SEED, NEW_SEED)

# 启动并行池（放在 rng 之后）
OLD_RNG = "rng(SEED, 'twister');              % 固定全局随机流，保证同一 SEED 可复现"
NEW_RNG = (OLD_RNG + "\n"
           "if isempty(gcp('nocreate')), parpool('Processes', NW); end   "
           "% 仅并行评估，不改变算法")
if s.count(OLD_RNG) != 1:
    print("[FAIL] rng line", s.count(OLD_RNG))
    sys.exit(1)
s = s.replace(OLD_RNG, NEW_RNG)

L1_OLD = "for i = 1:pop_size\n    % 1. 解码：获取索引"
L1_NEW = "parfor i = 1:pop_size\n    % 1. 解码：获取索引"
L2_OLD = "for i = 1:pop_size\n        n_a = population(i, 1);"
L2_NEW = "parfor i = 1:pop_size\n        n_a = population(i, 1);"

for old, new in ((L1_OLD, L1_NEW), (L2_OLD, L2_NEW)):
    if s.count(old) != 1:
        print("[FAIL] loop", s.count(old), repr(old[:40]))
        sys.exit(1)
    s = s.replace(old, new)

io.open(OUT, "w", encoding="utf-8").write(s)
print("[written]", OUT)

# 冒烟副本：3 代，仅测速
SMOKE_DIR.mkdir(exist_ok=True)
t = s.replace("max_generations = 1500;", "max_generations = 3;   % SMOKE TEST ONLY")
if "SMOKE TEST ONLY" not in t:
    print("[FAIL] max_generations not found")
    sys.exit(1)
io.open(SMOKE_DIR / "GA_smoke_par.m", "w", encoding="utf-8").write(t)
print("[written]", SMOKE_DIR / "GA_smoke_par.m")
