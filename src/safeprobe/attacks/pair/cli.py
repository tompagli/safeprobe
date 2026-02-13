"""PAIR CLI."""
import argparse
from safeprobe.config import load_config
from safeprobe.attacks.pair.attack import PAIRAttack
def main():
    p = argparse.ArgumentParser(description="SafeProbe PAIR Attack")
    p.add_argument("--attack-model", default="gpt-4o")
    p.add_argument("--target-model", required=True)
    p.add_argument("--goal", required=True)
    p.add_argument("--target-str", required=True)
    p.add_argument("--n-streams", type=int, default=3)
    p.add_argument("--n-iterations", type=int, default=3)
    p.add_argument("--output", default="pair_results.json")
    args = p.parse_args()
    config = load_config()
    r = PAIRAttack(config).execute(vars(args)|{"attack_model_type":"openai","target_model_type":"openai","output_file":args.output})
    print(f"{'ok' if r.get('success') else 'fail'}: {r.get('summary','')}")
if __name__=="__main__": main()
