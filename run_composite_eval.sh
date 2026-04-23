#!/usr/bin/env bash
# run_eval.sh
# Runs the full attack + multi-judge pipeline for any model config.
# Attacks are read from the 'attacks:' list in the YAML — no hardcoding.
#
# Prerequisites:
#   export OPENAI_API_KEY="sk-..."   # required for OpenAI targets/PAIR attacker
#   export HF_TOKEN="hf_..."         # required for gated HF models (Llama, LlamaGuard)
#   source venv/bin/activate
#
# Usage:
#   bash run_composite_eval.sh                        # run all default models
#   bash run_composite_eval.sh gpt4 llama qwen_full   # run specific configs

set -euo pipefail

CONFIGS=(gpt4 gpt5 llama qwen mistral)

# If arguments are passed, run only those models
if [[ $# -gt 0 ]]; then
    CONFIGS=("$@")
fi

for MODEL in "${CONFIGS[@]}"; do
    CONFIG="configs/${MODEL}.yaml"

    if [[ ! -f "$CONFIG" ]]; then
        echo "[SKIP] Config not found: $CONFIG"
        continue
    fi

    # Read attack list from YAML for display (safeprobe reads it internally)
    ATTACKS=$(awk '/^attacks:/{found=1; next} found && /^ *- [a-z]/{print $2} found && !/^ *-/{found=0}' "$CONFIG" | tr '\n' ',' | sed 's/,$//')

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Model  : ${MODEL}"
    echo "  Attacks: ${ATTACKS}"
    echo "══════════════════════════════════════════════════════"

    # Step 1 — Run all attacks listed in the config (no --attack flag = runs all)
    echo "[1/4] Running attacks: ${ATTACKS}..."
    safeprobe --config "$CONFIG" attack

    # Step 2 — Consolidate all attack JSON results into one CSV
    echo "[2/4] Consolidating results..."
    safeprobe --config "$CONFIG" consolidate

    # Step 3 — Multi-judge (sequential: one model in VRAM at a time)
    echo "[3/4] Running multi-judge + inter-rater agreement..."
    safeprobe --config "$CONFIG" multijudge

    # Step 4 — Generate report
    echo "[4/4] Generating report..."
    safeprobe --config "$CONFIG" report --use-judge

    echo "[DONE] ${MODEL} — results in: $(grep results_dir "$CONFIG" | awk '{print $2}')"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  All evaluations complete."
echo "  Results:"
for MODEL in "${CONFIGS[@]}"; do
    DIR=$(grep results_dir configs/${MODEL}.yaml 2>/dev/null | awk '{print $2}')
    [[ -n "$DIR" ]] && echo "    ${MODEL}: ${DIR}/"
done
echo "══════════════════════════════════════════════════════"
