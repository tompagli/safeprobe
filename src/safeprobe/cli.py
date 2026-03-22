"""SafeProbe CLI entry point."""
import sys, argparse, json, glob, os
from pathlib import Path
from safeprobe.config import Config, load_config

BANNER = "SafeProbe v1.0.0 - LLM Safety Alignment Evaluation Toolkit"


def _build_parser():
    p = argparse.ArgumentParser(prog="safeprobe", description="SafeProbe")
    p.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    p.add_argument("--target-model", type=str)
    p.add_argument("--target-model-type", choices=["openai", "anthropic", "google", "ollama", "xai"])
    p.add_argument("--judge-model", type=str)
    p.add_argument("--judge-model-type", type=str)
    p.add_argument("--iterations", type=int)
    p.add_argument("--results-dir", type=str)
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="command")

    # attack subcommand
    atk = sub.add_parser("attack", help="Run adversarial attacks")
    atk.add_argument("--attack", choices=["promptmap", "pair", "cipherchat", "all"], default="all")

    # judge subcommand
    jdg = sub.add_parser("judge", help="Run CoT judge on consolidated CSV")
    jdg.add_argument("--csv", type=str, help="Path to consolidated CSV (default: results/consolidated.csv)")
    jdg.add_argument("--output", type=str, help="Output CSV path (default: overwrites input)")
    jdg.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")

    # consolidate subcommand
    con = sub.add_parser("consolidate", help="Consolidate JSON results into CSV")
    con.add_argument("--output", type=str, help="Output CSV path")

    # report subcommand
    rpt = sub.add_parser("report", help="Generate evaluation report")
    rpt.add_argument("--csv", type=str, help="Path to consolidated CSV")
    rpt.add_argument("--output-dir", type=str, default="reports")
    rpt.add_argument("--use-judge", action="store_true", help="Use judge results for metrics")

    # pipeline subcommand (attack -> consolidate -> judge -> report)
    pip = sub.add_parser("pipeline", help="Run full pipeline: attack -> consolidate -> judge -> report")
    pip.add_argument("--attack", choices=["promptmap", "pair", "cipherchat", "all"], default="all")
    pip.add_argument("--skip-judge", action="store_true", help="Skip the judge step")

    return p


def _apply_config_overrides(config, args):
    if args.target_model:
        config.target_model = args.target_model
    if args.target_model_type:
        config.target_model_type = args.target_model_type
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.judge_model_type:
        config.judge_model_type = args.judge_model_type
    if args.iterations:
        config.iterations = args.iterations
    if args.results_dir:
        config.results_dir = Path(args.results_dir)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    print(BANNER)

    config = load_config(args.config)
    _apply_config_overrides(config, args)

    if not config.validate():
        print("Error: No API keys configured. Set OPENAI_API_KEY or similar.")
        sys.exit(1)

    if args.command == "attack":
        _cmd_attack(config, args.attack)
    elif args.command == "judge":
        _cmd_judge(config, args)
    elif args.command == "consolidate":
        _cmd_consolidate(config, args)
    elif args.command == "report":
        _cmd_report(config, args)
    elif args.command == "pipeline":
        _cmd_pipeline(config, args)
    else:
        # No subcommand — backwards compat: treat --target-model as attack
        if args.target_model:
            _cmd_attack(config, "all")
        else:
            parser.print_help()


# ── Attack ────────────────────────────────────────────────────────────────────

def _cmd_attack(config, attack_name):
    from safeprobe.attacks.promptmap.attack import PromptMapAttack
    from safeprobe.attacks.pair.attack import PAIRAttack
    from safeprobe.attacks.cipherchat.attack import CipherChatAttack
    from safeprobe.datasets.adapters import create_adapter

    attacks_to_run = ["promptmap", "pair", "cipherchat"] if attack_name == "all" else [attack_name]

    for name in attacks_to_run:
        print(f"\n--- Running {name} ---")
        try:
            if name == "promptmap":
                atk = PromptMapAttack(config=config)
                api_key = config.get_api_key(config.target_model_type) or ""
                target = create_adapter(config.target_model, config.target_model_type, api_key=api_key)
                output_file = str(config.results_dir / "promptmap_results.json")
                r = atk.execute(
                    target_adapter=target,
                    iterations=getattr(config, "iterations", 3),
                    output_file=output_file,
                )
                total = r.get("metrics", {}).get("total_tests", 0)
                failed = r.get("metrics", {}).get("failed", 0)
                asr = r.get("metrics", {}).get("asr", 0)
                print(f"  PromptMap: {total} tests, {failed} failed, ASR={asr:.1f}%")

            elif name == "pair":
                atk = PAIRAttack(config)
                r = atk.execute(atk.get_default_parameters())
                if r.get("success"):
                    print(f"  PAIR: {r.get('summary', {})}")
                else:
                    print(f"  PAIR failed: {r.get('error', 'unknown error')}")

            elif name == "cipherchat":
                atk = CipherChatAttack(config)
                r = atk.execute(atk.get_default_parameters())
                if r.get("success"):
                    print(f"  CipherChat: {r.get('summary', {})}")
                else:
                    print(f"  CipherChat failed: {r.get('error', 'unknown error')}")

        except Exception as e:
            print(f"  {name} error: {e}")


# ── Consolidate ───────────────────────────────────────────────────────────────

def _cmd_consolidate(config, args=None):
    from safeprobe.analysis.consolidator import ResultsConsolidator

    results_dir = config.results_dir
    json_files = sorted(glob.glob(str(results_dir / "*.json")))

    if not json_files:
        print(f"No JSON files found in {results_dir}/")
        return None

    print(f"Consolidating {len(json_files)} JSON files from {results_dir}/")
    for f in json_files:
        print(f"  - {Path(f).name}")

    consolidator = ResultsConsolidator()
    df = consolidator.consolidate_from_json_files(json_files)

    output = args.output if args and hasattr(args, "output") and args.output else str(results_dir / "consolidated.csv")
    consolidator.save_csv(df, output)
    print(f"Consolidated {len(df)} entries -> {output}")
    return output


# ── Judge ─────────────────────────────────────────────────────────────────────

def _cmd_judge(config, args=None):
    from safeprobe.analysis.judge import JailbreakJudge

    csv_path = None
    if args and hasattr(args, "csv") and args.csv:
        csv_path = args.csv
    else:
        csv_path = str(config.results_dir / "consolidated.csv")

    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        print("Run 'safeprobe consolidate' first, or pass --csv path/to/file.csv")
        return None

    output = args.output if args and hasattr(args, "output") and args.output else csv_path
    delay = args.delay if args and hasattr(args, "delay") else 0.5

    print(f"\n--- Running CoT Judge ---")
    print(f"  Model: {config.judge_model} ({config.judge_model_type})")
    print(f"  Input: {csv_path}")
    print(f"  Output: {output}")

    judge = JailbreakJudge(config)
    df = judge.judge_csv(csv_path, output_path=output, delay=delay)

    # Print summary
    if "judge_result" in df.columns:
        counts = df["judge_result"].value_counts()
        print(f"\n  Judge results:")
        for label, count in counts.items():
            print(f"    {label}: {count}")

    return output


# ── Report ────────────────────────────────────────────────────────────────────

def _cmd_report(config, args=None):
    from safeprobe.analysis.consolidator import ResultsConsolidator
    from safeprobe.analysis.report_gen import ReportGenerator

    csv_path = None
    if args and hasattr(args, "csv") and args.csv:
        csv_path = args.csv
    else:
        csv_path = str(config.results_dir / "consolidated.csv")

    if not Path(csv_path).exists():
        print(f"CSV not found: {csv_path}")
        return

    use_judge = args.use_judge if args and hasattr(args, "use_judge") else False
    output_dir = args.output_dir if args and hasattr(args, "output_dir") else "reports"

    import pandas as pd
    df = pd.read_csv(csv_path, encoding="utf-8")
    consolidator = ResultsConsolidator()

    print(f"\n--- Evaluation Report ---")
    print(f"  Total entries: {len(df)}")

    asr_metrics = consolidator.compute_asr(df, use_judge=use_judge)
    if "overall" in asr_metrics:
        ov = asr_metrics["overall"]
        print(f"  Overall ASR: {ov['asr']:.1f}% ({ov['successful']}/{ov['total']})")
    if "by_technique" in asr_metrics:
        for tool, m in asr_metrics["by_technique"].items():
            print(f"    {tool}: ASR={m['asr']:.1f}% ({m['successful']}/{m['total']})")

    robustness = consolidator.compute_robustness_score(df, use_judge=use_judge)
    print(f"  Robustness Score: {robustness:.1f}%")

    # Generate report files
    os.makedirs(output_dir, exist_ok=True)
    report_gen = ReportGenerator()

    # Add categories
    df["attack_category"] = df["attack_prompt"].apply(report_gen.categorize_attack)

    # Text report
    asr_metrics["robustness_score"] = robustness
    report_text = report_gen.generate_text_report(df, asr_metrics,
                                                   output_path=os.path.join(output_dir, "safeprobe_report.txt"))
    # JSON report
    report_gen.generate_json_report(df, asr_metrics,
                                     output_path=os.path.join(output_dir, "safeprobe_report.json"))
    # PDF report (optional, needs matplotlib)
    try:
        report_gen.generate_pdf_report(df, asr_metrics,
                                        output_path=os.path.join(output_dir, "safeprobe_report.pdf"))
    except Exception as e:
        logger.warning(f"PDF generation skipped: {e}")

    print(f"\n  Reports saved to {output_dir}/")


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def _cmd_pipeline(config, args):
    attack_name = args.attack if args and hasattr(args, "attack") else "all"
    skip_judge = args.skip_judge if args and hasattr(args, "skip_judge") else False

    # Step 1: Run attacks
    print("\n" + "=" * 50)
    print("STEP 1: Running attacks")
    print("=" * 50)
    _cmd_attack(config, attack_name)

    # Step 2: Consolidate
    print("\n" + "=" * 50)
    print("STEP 2: Consolidating results")
    print("=" * 50)
    csv_path = _cmd_consolidate(config)
    if not csv_path:
        print("No results to consolidate. Stopping.")
        return

    # Step 3: Judge (optional)
    if not skip_judge:
        print("\n" + "=" * 50)
        print("STEP 3: Running CoT Judge")
        print("=" * 50)

        class _JudgeArgs:
            csv = csv_path
            output = csv_path
            delay = 0.5

        _cmd_judge(config, _JudgeArgs())

    # Step 4: Report
    print("\n" + "=" * 50)
    print(f"STEP 4: Generating report")
    print("=" * 50)

    class _ReportArgs:
        csv = csv_path
        output_dir = "reports"
        use_judge = not skip_judge

    _cmd_report(config, _ReportArgs())

    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()