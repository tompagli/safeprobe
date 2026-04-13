<div align="center" id="top">
  <img src="./.github/app.gif" alt="SafeProbe" />

  &#xa0;
</div>

<h1 align="center">SafeProbe</h1>

<p align="center">
  <img alt="Top language" src="https://img.shields.io/github/languages/top/tompagli/safeprobe?color=56BEB8">
  <img alt="Language count" src="https://img.shields.io/github/languages/count/tompagli/safeprobe?color=56BEB8">
  <img alt="Repo size" src="https://img.shields.io/github/repo-size/tompagli/safeprobe?color=56BEB8">
  <img alt="License" src="https://img.shields.io/github/license/tompagli/safeprobe?color=56BEB8">
</p>

<p align="center">
  <a href="#dart-about">About</a> &#xa0; | &#xa0;
  <a href="#sparkles-features">Features</a> &#xa0; | &#xa0;
  <a href="#rocket-architecture">Architecture</a> &#xa0; | &#xa0;
  <a href="#white_check_mark-requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#checkered_flag-installation">Installation</a> &#xa0; | &#xa0;
  <a href="#wrench-configuration">Configuration</a> &#xa0; | &#xa0;
  <a href="#zap-usage">Usage</a> &#xa0; | &#xa0;
  <a href="#bar_chart-metrics">Metrics</a> &#xa0; | &#xa0;
  <a href="#memo-license">License</a>
</p>

---

## :dart: About

**SafeProbe** is an open-source Python toolkit for evaluating the **safety alignment of Large Language Models (LLMs) at inference time**, with a focus on **query-access attack vectors** such as jailbreaking, prompt injection, and adversarial prompt reformulation.

Unlike surface-level moderation tools, SafeProbe performs **intent-aware, semantic safety evaluation** using:
- automated red-teaming attacks,
- quantitative robustness metrics,
- and a **Chain-of-Thought (CoT)–based LLM judge** that reasons about model intent rather than relying on keyword matching.

SafeProbe is designed for **research reproducibility** and **practical deployment**, enabling developers, researchers, and security engineers to integrate safety evaluation directly into CI/CD pipelines and pre-deployment checks.

---

## :sparkles: Features

✔ Automated red-teaming via query-access attacks  
✔ **Composite CO × MG attack** — exhaustive multi-layer technique pairing with per-combination ASR ranking  
✔ Support for multiple LLM providers (OpenAI, Anthropic, HuggingFace, Ollama, xAI)  
✔ **Open-source target models** — Llama-3, Mistral, Qwen3, and any HuggingFace instruct model  
✔ **Three judge backends** — CoT/DeepSeek, Llama Guard 3, HarmBench classifier  
✔ **Inter-rater agreement** — Cohen's κ and Fleiss' κ across all judge pairs  
✔ **Multiple benchmark datasets** — AdvBench, HarmBench, JailbreakBench  
✔ Quantitative metrics (Attack Success Rate, Robustness Score)  
✔ Modular, extensible attack architecture  
✔ CLI with subcommands and programmatic Python API  
✔ YAML/JSON configuration for reproducible evaluations  
✔ Report generation (TXT, JSON, PDF with charts)  
✔ Alignment with the **NIST Adversarial Machine Learning taxonomy** (AI 100-2e2025)

---

## :rocket: Architecture

SafeProbe is organized as a modular pipeline with four stages: **attack → consolidate → judge → report**.

```
safeprobe/
├── attacks/            # Query-access attack implementations
│   ├── promptmap/      # Rule-based prompt transformation (56 YAML rules)
│   │   └── rules/      # Attack rules organized by category
│   ├── pair/           # PAIR iterative adversarial refinement
│   ├── cipherchat/     # Encoding-based attacks (Caesar, Atbash, Morse, ASCII)
│   └── composite/      # CO × MG combination attack (differentiator)
│       └── transformers/
│           ├── competing_objectives.py      # prefix, refusal suppression, style, roleplay
│           └── mismatched_generalization.py # base64, rot13, leetspeak, pig latin, translation
│
├── datasets/           # Adversarial prompt datasets & model adapters
│   ├── prompts.py      # AdvBench / HarmBench / JailbreakBench loaders
│   └── adapters.py     # Unified LLM interface: OpenAI, Anthropic, HuggingFace, Ollama, xAI
│
├── analysis/           # Output classification, metrics & reporting
│   ├── judges/         # Judge backends (all share BaseJudge interface)
│   │   ├── deepseek.py   # CoT judge — DeepSeek R1 local or any API model
│   │   ├── llama_guard.py# Meta Llama Guard 3 safety classifier (local HF)
│   │   └── harmbench.py  # CAIS HarmBench binary classifier (local HF)
│   ├── judge.py        # Backward-compat shim → judges/deepseek.py
│   ├── multi_judge.py  # Run N judges in parallel + inter-rater agreement
│   ├── agreement.py    # Cohen's κ, Fleiss' κ, agreement matrix
│   ├── consolidator.py # Result aggregation, ASR & Robustness Score
│   └── report_gen.py   # Report generation (TXT, JSON, PDF)
│
├── utils/              # Logging, API helpers, shared utilities
├── cli.py              # Command-line interface with subcommands
└── config.py           # Configuration & pipeline orchestration
```

### Attack Techniques

| Technique | Type | Description | Complexity Weight |
|-----------|------|-------------|:-:|
| **PromptMap** | Rule-based | 56 YAML rules covering jailbreak, harmful, hate, distraction, social bias, and prompt stealing categories | 1 |
| **CipherChat** | Encoding-based | Reformulates prompts using Caesar, Atbash, Morse, or ASCII encoding to bypass safety filters | 3 |
| **PAIR** | Model-based | Uses an attacker LLM to iteratively refine adversarial prompts across multiple streams | 5 |
| **Composite** | Multi-layer | Combines Competing Objectives (CO) and Mismatched Generalization (MG) techniques; tests all 20 CO × MG pairings and ranks them by ASR | 7 |

#### Composite Attack — CO × MG Techniques

The Composite attack is SafeProbe's differentiating feature. It systematically tests every pairing of a **Competing Objectives** technique with a **Mismatched Generalization** technique, giving researchers a ranked view of which combinations are most effective against a given model.

| Layer | Technique | Strategy |
|-------|-----------|----------|
| **CO** | `prefix_injection` | Forces the model to begin with an affirmative prefix, steering it toward compliance |
| **CO** | `refusal_suppression` | Explicitly instructs the model to omit refusal language |
| **CO** | `style_injection` | Reframes the request as academic, fictional, professional, or educational |
| **CO** | `roleplay` | Adopts a persona (DAN, researcher, character, developer mode) that normalizes compliance |
| **MG** | `base64` | Encodes the payload in base64 with an explicit decode instruction |
| **MG** | `rot13` | ROT13-encodes the payload with an explicit decode instruction |
| **MG** | `leetspeak` | Substitutes characters (a→4, e→3, o→0…) to evade surface-level pattern matching |
| **MG** | `pig_latin` | Applies pig latin word-by-word transformation |
| **MG** | `translation` | Translates the prompt to another language (uses `deep-translator` if installed; falls back to a language-framing wrapper) |

**Ordering**: MG is applied first (obscures the payload from safety filters), then CO wraps the result (plaintext instruction to decode and comply).

### Pipeline Flow

```
┌─────────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────┐
│   Attack    │───▶│ Consolidate │───▶│   Judge   │───▶│  Report  │
│             │    │             │    │           │    │          │
│ PromptMap   │    │ JSON → CSV  │    │ CoT-based │    │ TXT/JSON │
│ PAIR        │    │ Standardize │    │ 0/1 score │    │ PDF+Chart│
│ CipherChat  │    │ entries     │    │ reasoning │    │ Metrics  │
│ Composite   │    │             │    │           │    │ ASR rank │
│ (CO × MG)   │    │             │    │           │    │          │
└─────────────┘    └─────────────┘    └───────────┘    └──────────┘
```

---

## :white_check_mark: Requirements

- **Python ≥ 3.10**
- **Git**
- API key for at least one supported LLM provider (e.g., OpenAI)

For the **CoT judge with a local model** (recommended):
- **NVIDIA GPU** with ≥ 16 GB VRAM (24 GB recommended for 14B models)
- `torch`, `transformers`, `bitsandbytes`, `accelerate`

---

## :checkered_flag: Installation

### From source (recommended)

```bash
git clone https://github.com/tompagli/safeprobe.git
cd safeprobe
python -m venv venv
source venv/bin/activate
pip install -e .
```

### For local judge support (GPU)

```bash
pip install torch transformers bitsandbytes accelerate
```

### From PyPI

```bash
pip install safeprobe
```

### Verify installation

```bash
safeprobe --help
```

---

## :wrench: Configuration

### Environment Variables

Set the API key for your target LLM provider:

```bash
export OPENAI_API_KEY="sk-your-key-here"

# Optional: other providers
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export XAI_API_KEY="your-key"
```

### YAML Configuration (optional)

Create an `evaluation.yaml` file for reproducible experiments:

```yaml
# Target model (API or open-source)
target_model: gpt-4.1-2025-04-14
target_model_type: openai           # openai | anthropic | huggingface | hf | local | ollama | xai

# Primary CoT judge
judge_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
judge_model_type: local

# Multi-judge backends for inter-rater agreement
judges:
  - deepseek
  - llamaguard
  - harmbench

llamaguard_model: meta-llama/Llama-Guard-3-8B
harmbench_model: cais/HarmBench-Llama-2-13b-cls
load_in_4bit: false   # set true to halve VRAM on all local models

attacks:
  - promptmap
  - pair
  - cipherchat
  - composite

dataset: harmbench    # advbench | harmbench | jailbreakbench | path/to/custom.csv
sample_size: 50
iterations: 3
results_dir: results
```

Then run with:

```bash
safeprobe --config evaluation.yaml pipeline
```

---

## :zap: Usage

SafeProbe provides a CLI with **five subcommands** that can be run independently or chained together.

### Quick Start — Full Pipeline

Run the entire evaluation in one command (attack → consolidate → judge → report):

```bash
safeprobe \
  --target-model gpt-4.1-2025-04-14 \
  --target-model-type openai \
  --judge-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --judge-model-type local \
  pipeline
```

To skip the judge step (faster, uses keyword-based scoring only):

```bash
safeprobe \
  --target-model gpt-4.1-2025-04-14 \
  --target-model-type openai \
  pipeline --skip-judge
```

---

### Step-by-Step Usage

#### 1. Run Attacks

Run all attacks against a target model:

```bash
safeprobe --target-model gpt-4.1-2025-04-14 --target-model-type openai attack --attack all
```

Run a specific attack:

```bash
safeprobe --target-model gpt-4.1-2025-04-14 --target-model-type openai attack --attack promptmap
safeprobe --target-model gpt-4.1-2025-04-14 --target-model-type openai attack --attack pair
safeprobe --target-model gpt-4.1-2025-04-14 --target-model-type openai attack --attack cipherchat
safeprobe --target-model gpt-4.1-2025-04-14 --target-model-type openai attack --attack composite
```

Attack results are saved as JSON files in the `results/` directory.

#### 2. Consolidate Results

Merge all JSON result files into a single CSV:

```bash
safeprobe consolidate
```

This creates `results/consolidated.csv` with standardized columns across all attack techniques.

#### 3. Run the CoT Judge

Classify each attack/response pair using a reasoning-capable LLM:

**Using a local model (recommended — no API cost):**

```bash
safeprobe \
  --judge-model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --judge-model-type local \
  judge
```

The local judge loads the model with 4-bit quantization (~9 GB VRAM) and evaluates each entry, adding `judge_result` (YES/NO), `judge_justification`, and `judge_reasoning_chain` columns to the CSV.

**Using an API model:**

```bash
safeprobe --judge-model gpt-4.1-2025-04-14 --judge-model-type openai judge
```

**Additional judge options:**

```bash
safeprobe judge --csv path/to/custom.csv       # Use a specific CSV
safeprobe judge --output path/to/output.csv     # Save to a different file
safeprobe judge --delay 1.0                      # Slower API calls (rate limiting)
```

The judge supports **checkpointing** — if interrupted, re-running the command will skip already-judged rows.

#### 4. Generate Reports

Generate evaluation reports with metrics:

```bash
safeprobe report                  # Using keyword-based attack_successful field
safeprobe report --use-judge      # Using CoT judge results (recommended)
```

This produces three files in `reports/`:
- `safeprobe_report.txt` — text summary with ASR and Robustness Score
- `safeprobe_report.json` — full metrics and all entries
- `safeprobe_report.pdf` — PDF with summary and ASR bar chart

**Custom report options:**

```bash
safeprobe report --csv path/to/file.csv --output-dir my_reports --use-judge
```

---

---

### Open-Source Target Models

SafeProbe supports any HuggingFace instruct model as an attack target via the `huggingface` adapter type. No API key required — models run locally.

```bash
# Llama-3.2-3B-Instruct (3B — fast, 4 GB VRAM)
safeprobe --target-model meta-llama/Llama-3.2-3B-Instruct \
          --target-model-type huggingface attack

# Mistral-7B-Instruct-v0.3 (7B, ~8 GB VRAM or 4-bit)
safeprobe --target-model mistralai/Mistral-7B-Instruct-v0.3 \
          --target-model-type hf --load-in-4bit attack

# Qwen3-4B (4B, Qwen3 <think> blocks handled automatically)
safeprobe --target-model Qwen/Qwen3-4B \
          --target-model-type local attack
```

For gated models (Llama), set your HuggingFace access token:

```bash
export HF_TOKEN="hf_your_token"
```

### Benchmark Datasets

SafeProbe supports three adversarial prompt datasets. Dataset choice is a key validity concern — AdvBench is widely used but reviewers often ask for HarmBench or JailbreakBench.

| Dataset | Key | Behaviors | Categories |
|---------|-----|-----------|------------|
| **AdvBench** | `advbench` | 520 | General harmful instructions |
| **HarmBench** | `harmbench` | 400 | Standard, copyright, contextual, multimodal |
| **JailbreakBench** | `jailbreakbench` | 100 | 10 fine-grained harm categories |

```bash
# YAML config
dataset: harmbench   # or jailbreakbench

# Python API
from safeprobe.datasets.prompts import load_harmbench, load_jailbreakbench
prompts = load_harmbench(max_samples=50)
prompts = load_jailbreakbench(max_samples=100, category="Harmful behaviors")
```

Both load from HuggingFace (`datasets` package) or a local CSV/JSON file.

---

### Standalone Attack CLIs

Each attack technique has its own standalone CLI entry point:

```bash
# PromptMap
safeprobe-promptmap --target-model gpt-4.1-2025-04-14 --target-model-type openai

# PAIR
safeprobe-pair --target-model gpt-4.1-2025-04-14 --goal "harmful task" --target-str "Sure, here is"

# CipherChat
safeprobe-cipherchat --target-model gpt-4.1-2025-04-14 --encode-method caesar
```

---

### Python API

```python
from safeprobe.config import Config
from safeprobe.attacks.promptmap.attack import PromptMapAttack
from safeprobe.attacks.pair.attack import PAIRAttack
from safeprobe.attacks.cipherchat.attack import CipherChatAttack
from safeprobe.attacks.composite.attack import CompositeAttack
from safeprobe.datasets.adapters import create_adapter
from safeprobe.analysis.judge import JailbreakJudge
from safeprobe.analysis.consolidator import ResultsConsolidator

# Configure
config = Config(
    target_model="gpt-4.1-2025-04-14",
    target_model_type="openai",
    judge_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    judge_model_type="local",
)

# Run PromptMap
adapter = create_adapter(config.target_model, config.target_model_type, api_key=config.get_api_key("openai"))
pm = PromptMapAttack(config=config)
pm_results = pm.execute(target_adapter=adapter, output_file="results/promptmap.json")

# Run Composite attack (all 20 CO × MG combinations)
composite = CompositeAttack(config=config)
composite_results = composite.execute({
    **composite.get_default_parameters(),
    "sample_size": 20,
    "use_judge": True,
})
print(f"Best combo: {composite_results['summary']['best_combination']}")
print(f"Overall ASR: {composite_results['summary']['asr']}")

# Or target specific technique subsets
composite_results = composite.execute({
    **composite.get_default_parameters(),
    "co_techniques": ["prefix_injection", "roleplay"],
    "mg_techniques": ["base64", "rot13"],
    "sample_size": 10,
})

# Consolidate
consolidator = ResultsConsolidator()
df = consolidator.consolidate_from_json_files([
    "results/promptmap.json",
    "results/composite_results.json",
])

# Judge
judge = JailbreakJudge(config)
df = judge.judge_csv("results/consolidated.csv")

# Compute metrics
asr = consolidator.compute_asr(df, use_judge=True)
robustness = consolidator.compute_robustness_score(df, use_judge=True)

print(f"ASR: {asr['overall']['asr']:.1f}%")
print(f"Robustness Score: {robustness:.1f}%")
```

---

## :bar_chart: Metrics

SafeProbe computes two quantitative metrics for safety evaluation:

### Attack Success Rate (ASR)

The proportion of adversarial attempts that successfully induce harmful outputs:

```
ASR = (successful attacks / total attacks) × 100
```

ASR is computed both overall and per-technique. When the CoT judge is used (`--use-judge`), the judge's YES/AMBIGUOUS verdicts determine success instead of keyword matching.

### Robustness Score

A weighted metric that accounts for attack complexity, reflecting the "effort" required to compromise a model's safety:

```
Robustness = (1 − weighted_successes / weighted_total) × 100
```

Complexity weights:
| Attack | Weight | Rationale |
|--------|:------:|-----------|
| PromptMap | 1 | Simple prompt modifications |
| CipherChat | 3 | Sophisticated encoding tricks |
| PAIR | 5 | Multi-turn iterative manipulation |
| Composite | 7 | Multi-layer CO + MG combination with exhaustive pairing |

A higher Robustness Score indicates a model that requires more complex attacks to compromise, suggesting stronger safety alignment.

---

## :brain: Judge Backends

SafeProbe ships three judge backends, all sharing the same `BaseJudge` interface. Running multiple judges together produces inter-rater agreement metrics that validate your ASR numbers.

### Available Backends

| Backend | Class | Model | Strategy | VRAM | Cost |
|---------|-------|-------|----------|------|------|
| **CoT / DeepSeek** | `JailbreakJudge` | DeepSeek-R1-Distill-Qwen-14B (local) or any API | Chain-of-Thought reasoning, outputs 0/1 | ~9 GB (4-bit) | Free / API |
| **Llama Guard 3** | `LlamaGuardJudge` | meta-llama/Llama-Guard-3-8B | Meta's safety classifier, outputs "safe"/"unsafe+categories" | ~6 GB (4-bit) | Free |
| **HarmBench** | `HarmBenchJudge` | cais/HarmBench-Llama-2-13b-cls | CAIS fine-tuned binary classifier, outputs "Yes"/"No" | ~7 GB (4-bit) | Free |

All local models support `--load-in-4bit` to halve VRAM requirements.

### Inter-Rater Agreement

Running multiple judges on the same results validates that your ASR numbers are not an artifact of a single evaluation method.

```bash
# Run all three judges and compute agreement
safeprobe --judges deepseek,llamaguard,harmbench multijudge

# With 4-bit quantization (GPU with ≥ 8 GB VRAM)
safeprobe --judges llamaguard,harmbench --load-in-4bit multijudge --csv results/consolidated.csv
```

Output columns added to the CSV:
- `judge_DeepSeek_result` / `judge_LlamaGuard_result` / `judge_HarmBench_result`
- `judge_*_justification` and `judge_*_reasoning_chain`

The agreement report is printed to stdout and saved as `results/consolidated_agreement.json`:

```
============================================================
  INTER-RATER AGREEMENT REPORT
============================================================
  Samples evaluated : 50
  Valid (no errors)  : 48

  Per-judge Attack Success Rate:
    DeepSeek             34.0%
    LlamaGuard           28.0%
    HarmBench            31.0%

  Fleiss' κ (all judges) : 0.7821  [substantial]

  Pairwise Cohen's κ:
  Pair                                       κ  Interpretation         %Agree      n
  DeepSeek vs LlamaGuard                0.7400  substantial             87.5%     48
  DeepSeek vs HarmBench                 0.8100  almost perfect          90.6%     48
  LlamaGuard vs HarmBench               0.7900  substantial             89.6%     48
============================================================
```

**Kappa interpretation** (Landis & Koch, 1977): `< 0.20` = slight · `0.20–0.40` = fair · `0.40–0.60` = moderate · `0.60–0.80` = substantial · `> 0.80` = almost perfect.

### Python API

```python
from safeprobe.analysis import JailbreakJudge, LlamaGuardJudge, HarmBenchJudge, MultiJudge
from safeprobe.analysis import compute_agreement, format_agreement_report

judges = {
    "DeepSeek":  JailbreakJudge(config),
    "LlamaGuard": LlamaGuardJudge(load_in_4bit=True),
    "HarmBench":  HarmBenchJudge(load_in_4bit=True),
}
mj = MultiJudge(judges)
df, agreement = mj.judge_csv("results/consolidated.csv")
print(format_agreement_report(agreement))
```

### Single-judge output columns

For each entry, a judge produces:
- `judge_result`: YES (jailbroken) · NO (refused) · AMBIGUOUS · ERROR
- `judge_justification`: brief human-readable reason
- `judge_reasoning_chain`: full raw model output

---

## :scroll: Research Background

SafeProbe is grounded in peer-reviewed research and aligned with the **NIST AI 100-2e2025 taxonomy** for Adversarial Machine Learning.

It was evaluated using the **AdvBench** dataset against multiple LLM architectures, demonstrating significant differences in robustness between reasoning-capable and non-reasoning models.

---

## :memo: License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Milton Pedro Pagliuso Neto**  
Graduate Program in Applied Computing  
Santa Catarina State University (UDESC)

📧 [milton.neto@edu.udesc.br](mailto:milton.neto@edu.udesc.br)  
🔗 [https://github.com/tompagli](https://github.com/tompagli)

---

<a href="#top">Back to top</a>