"""nanoGCG CLI."""
import argparse
from safeprobe.config import load_config
from safeprobe.attacks.nanoGCG.attack import nanoGCGAttack
def main():
    p = argparse.ArgumentParser(description="SafeProbe nanoGCG Attack")
    p.add_argument("--target-model", required=True)
    p.add_argument("--target-model-type", default="openai")
    p.add_argument("--attacker-model", default=nanoGCGAttack.DEFAULT_ATTACKER_MODEL)
    p.add_argument("--gcg-steps", type=int, default=50)
    p.add_argument("--gcg-search-width", type=int, default=64)
    p.add_argument("--gcg-topk", type=int, default=64)
    p.add_argument("--target-prefix", default="Sure, here is")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", default="nanogcg_results.json")
    args = p.parse_args()
    r = nanoGCGAttack(load_config()).execute({
        "target_model": args.target_model, "target_model_type": args.target_model_type,
        "attacker_model": args.attacker_model, "gcg_steps": args.gcg_steps,
        "gcg_search_width": args.gcg_search_width, "gcg_topk": args.gcg_topk,
        "target_prefix": args.target_prefix, "iterations": args.iterations,
        "output_file": args.output,
    })
    print(f"{'ok' if r.get('success') else 'fail'}: {r.get('summary', r.get('error', ''))}")
if __name__=="__main__": main()