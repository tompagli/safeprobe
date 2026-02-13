"""SafeProbe CLI entry point."""
import sys, argparse, json
from pathlib import Path
from safeprobe.config import Config, load_config

BANNER = "SafeProbe v1.0.0 - LLM Safety Alignment Evaluation Toolkit"

def _build_parser():
    p = argparse.ArgumentParser(prog="safeprobe", description="SafeProbe")
    p.add_argument("--config", type=str)
    p.add_argument("--attack", choices=["promptmap","pair","cipherchat","all"], default="all")
    p.add_argument("--target-model", type=str)
    p.add_argument("--target-model-type", choices=["openai","anthropic","google","ollama","xai"])
    p.add_argument("--judge-model", type=str)
    p.add_argument("--iterations", type=int)
    p.add_argument("--results-dir", type=str)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("-v","--verbose", action="store_true")
    return p

def main():
    parser = _build_parser()
    args = parser.parse_args()
    print(BANNER)
    config = load_config(args.config)
    if args.target_model: config.target_model = args.target_model
    if args.target_model_type: config.target_model_type = args.target_model_type
    if args.judge_model: config.judge_model = args.judge_model
    if args.iterations: config.iterations = args.iterations
    if args.results_dir: config.results_dir = Path(args.results_dir)
    if not config.validate():
        print("No API keys configured."); sys.exit(1)
    if args.interactive or (not args.target_model and not args.config):
        _run_interactive(config)
    else:
        _run_pipeline(config, args.attack)

def _run_interactive(config):
    from safeprobe.attacks.promptmap.attack import PromptMapAttack
    from safeprobe.attacks.pair.attack import PAIRAttack
    from safeprobe.attacks.cipherchat.attack import CipherChatAttack
    results = []
    while True:
        print("\n=== SafeProbe ===\n1. PromptMap  2. PAIR  3. CipherChat  4. Summary  5. Save  6. Exit")
        c = input("Select: ").strip()
        if c in ("1","2","3"):
            cls = [PromptMapAttack, PAIRAttack, CipherChatAttack][int(c)-1]
            atk = cls(config)
            try:
                r = atk.execute(atk.get_parameters()); results.append(r)
            except Exception as e: print(f"Error: {e}")
        elif c == "4":
            for i, r in enumerate(results, 1): print(f"  {i}. {r.get('technique', 'unknown')} done")
        elif c == "5":
            with open("safeprobe_results.json","w") as f: json.dump(results, f, indent=2, default=str)
        elif c == "6": break

def _run_pipeline(config, attack_name):
    from safeprobe.attacks.promptmap.attack import PromptMapAttack
    from safeprobe.attacks.pair.attack import PAIRAttack
    from safeprobe.attacks.cipherchat.attack import CipherChatAttack
    m = {"promptmap": PromptMapAttack, "pair": PAIRAttack, "cipherchat": CipherChatAttack}
    for name in (list(m) if attack_name == "all" else [attack_name]):
        atk = m[name](config)
        r = atk.execute(atk.get_default_parameters())
        print(f"  {name}: done")

if __name__ == "__main__":
    main()
