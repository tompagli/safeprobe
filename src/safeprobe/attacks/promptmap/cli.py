"""PromptMap CLI."""
import argparse
from safeprobe.config import load_config
from safeprobe.attacks.promptmap.attack import PromptMapAttack
def main():
    p = argparse.ArgumentParser(description="SafeProbe PromptMap Attack")
    p.add_argument("--target-model", required=True)
    p.add_argument("--target-model-type", required=True, choices=["openai","anthropic","google","ollama","xai"])
    p.add_argument("--prompts", default="system-prompts.txt")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", default="promptmap_results.json")
    args = p.parse_args()
    config = load_config()
    config.target_model = args.target_model; config.target_model_type = args.target_model_type
    r = PromptMapAttack(config).execute({"target_model":args.target_model,"target_model_type":args.target_model_type,
        "prompts_path":args.prompts,"iterations":args.iterations,"output_file":args.output})
    print(f"{'ok' if r.get('success') else 'fail'}")
if __name__=="__main__": main()
