"""CipherChat CLI."""
import argparse
from safeprobe.config import load_config
from safeprobe.attacks.cipherchat.attack import CipherChatAttack
def main():
    p = argparse.ArgumentParser(description="SafeProbe CipherChat Attack")
    p.add_argument("--target-model", required=True)
    p.add_argument("--target-model-type", default="openai")
    p.add_argument("--encode-method", default="caesar", choices=["unchange","caesar","atbash","morse","ascii","baseline"])
    p.add_argument("--prompts", default="advbench")
    p.add_argument("--sample-size", type=int, default=10)
    p.add_argument("--output", default="cipherchat_results.json")
    args = p.parse_args()
    r = CipherChatAttack(load_config()).execute({"target_model":args.target_model,"target_model_type":args.target_model_type,
        "encode_method":args.encode_method,"prompts":args.prompts,"sample_size":args.sample_size,"output_file":args.output})
    print(f"{'ok' if r.get('success') else 'fail'}: {r.get('summary','')}")
if __name__=="__main__": main()
