# 顺序跑 3 个种子；每个种子 1500 代，20 workers。
# 只并行「种群评估」，算法与参数未改。每个种子的日志独立落盘。
$ErrorActionPreference = "Continue"
Set-Location "F:\CherryInAcademic\projects\20261019_iotj_revision\r2\ga_seeds"
$env:GA_WORKERS = "20"
$matlab = "G:\MATLAB2024b\bin\matlab.exe"

"[$(Get-Date -Format o)] start" | Out-File -Encoding utf8 run_all.log
foreach ($s in 1, 2, 3) {
    $env:GA_SEED = "$s"
    $log = "run_seed$s.log"
    "[$(Get-Date -Format o)] seed $s begin" | Out-File -Encoding utf8 -Append run_all.log
    & $matlab -batch "GA_r2_par" *> $log
    "[$(Get-Date -Format o)] seed $s end rc=$LASTEXITCODE" | Out-File -Encoding utf8 -Append run_all.log
}
"[$(Get-Date -Format o)] all done" | Out-File -Encoding utf8 -Append run_all.log
