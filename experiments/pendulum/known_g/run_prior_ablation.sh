#!/usr/bin/env bash
# 先验力度消融：同一数据、同一种子、同一 MCMC 配置，只改结构先验。
# 每个配置 20 条链（8 主 + 4 对照A + 4 对照B + 4 对照C），输出到
# pendulum_runs/<tag>/。单温度，与 2026-10-17 那批结果同口径。
set -u
cd "$(dirname "$0")/../../.."

PY=.venv/bin/python
LOGDIR=experiments/pendulum/known_g/pendulum_runs/_logs
mkdir -p "$LOGDIR"

run () {
    local prior="$1"; local scale="$2"
    local tag="${prior}"; [ "$scale" != "1" ] && tag="${prior}_x${scale}"
    echo "==== $(date +%T) start ${tag} ===="
    $PY experiments/pendulum/known_g/pendulum.py --prior "$prior" --scale "$scale" --no-fig \
        > "$LOGDIR/run_${tag}.log" 2>&1
    echo "==== $(date +%T) done  ${tag} (exit=$?) ===="
}

run bms_wiki_nv13_np13        0.5
run bms_wiki_nv13_np13        0.25
run bms_wiki_nv13_np13        2.0
run bms_wiki_nv13_np13_linear 1
echo "==== ALL DONE $(date +%T) ===="
