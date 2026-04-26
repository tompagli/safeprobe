#!/usr/bin/env bash
# run_eval.sh
# Runs the full attack + multi-judge pipeline for any model config.
# Attacks are read from the 'attacks:' list in the YAML — no hardcoding.
# Already-completed attack steps are skipped automatically.
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

    # Read results_dir and attack list from YAML
    RESULTS_DIR=$(awk '/^results_dir:/{print $2}' "$CONFIG")
    ATTACKS=$(awk '/^attacks:/{found=1; next} found && /^ *- [a-z]/{print $2} found && !/^ *-/{found=0}' "$CONFIG" | tr '\n' ',' | sed 's/,$//')

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Model     : ${MODEL}"
    echo "  Attacks   : ${ATTACKS}"
    echo "  Result dir: ${RESULTS_DIR}"
    echo "══════════════════════════════════════════════════════"

    # Step 1 — Run each attack individually, skipping ones with existing output files
    echo "[1/4] Checking attacks..."
    ATTACK_OUTPUT_MAP="promptmap:promptmap_results.json cipherchat:cipherchat_results.json pair:pair_results.json composite:composite_results.json"

    PENDING_ATTACKS=()
    for ENTRY in $ATTACK_OUTPUT_MAP; do
        ATK="${ENTRY%%:*}"
        FILE="${ENTRY##*:}"
        # Only process attacks listed in this config
        if echo "$ATTACKS" | grep -qw "$ATK"; then
            if [[ -f "${RESULTS_DIR}/${FILE}" ]]; then
                echo "  [SKIP] ${ATK} — ${RESULTS_DIR}/${FILE} already exists"
            else
                PENDING_ATTACKS+=("$ATK")
            fi
        fi
    done

    if [[ ${#PENDING_ATTACKS[@]} -eq 0 ]]; then
        echo "  [SKIP] All attacks already completed."
    else
        for ATK in "${PENDING_ATTACKS[@]}"; do
            echo "  [RUN] ${ATK}..."
            safeprobe --config "$CONFIG" attack --attack "$ATK"
        done
    fi

    # Step 2 — Consolidate all attack JSON results into one CSV
    echo "[2/4] Consolidating results..."
    safeprobe --config "$CONFIG" consolidate

    # Step 3 — Multi-judge (sequential: one model in VRAM at a time)
    echo "[3/4] Running multi-judge + inter-rater agreement..."
    safeprobe --config "$CONFIG" multijudge

    # Step 4 — Generate report
    echo "[4/4] Generating report..."
    safeprobe --config "$CONFIG" report --use-judge

    echo "[DONE] ${MODEL} — results in: ${RESULTS_DIR}/"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  All evaluations complete."
echo "  Results:"
for MODEL in "${CONFIGS[@]}"; do
    DIR=$(awk '/^results_dir:/{print $2}' "configs/${MODEL}.yaml" 2>/dev/null)
    [[ -n "$DIR" ]] && echo "    ${MODEL}: ${DIR}/"
done
echo "══════════════════════════════════════════════════════"
