#!/usr/bin/env bash
# run_composite_eval.sh
# Runs the full composite attack + multi-judge pipeline for all five models.
#
# Prerequisites:
#   export OPENAI_API_KEY="sk-..."
#   export HF_TOKEN="hf_..."          # required for Llama (gated model)
#   source venv/bin/activate
#
# Usage:
#   bash run_composite_eval.sh               # run all models
#   bash run_composite_eval.sh gpt4 llama    # run specific models

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

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Model: ${MODEL}"
    echo "══════════════════════════════════════════════════════"

    # Step 1 — Run composite attack
    echo "[1/4] Running composite attack..."
    safeprobe --config "$CONFIG" attack --attack composite

    # Step 2 — Consolidate JSON results to CSV
    echo "[2/4] Consolidating results..."
    safeprobe --config "$CONFIG" consolidate

    # Step 3 — Multi-judge (DeepSeek + LlamaGuard + HarmBench) + agreement
    echo "[3/4] Running multi-judge + inter-rater agreement..."
    safeprobe --config "$CONFIG" multijudge

    # Step 4 — Generate report using judge verdicts
    echo "[4/4] Generating report..."
    safeprobe --config "$CONFIG" report --use-judge

    echo "[DONE] ${MODEL} — results in: $(grep results_dir configs/${MODEL}.yaml | awk '{print $2}')"
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
