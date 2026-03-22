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
✔ Support for multiple LLM providers (OpenAI, Anthropic, Google, Ollama, xAI)  
✔ CoT-based semantic safety classification using local or API-based models  
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
│   └── cipherchat/     # Encoding-based attacks (Caesar, Atbash, Morse, ASCII)
│
├── datasets/           # Adversarial prompt datasets & model adapters
│   ├── prompts.py      # AdvBench loader (HuggingFace or local)
│   └── adapters.py     # Unified LLM interface for all providers
│
├── analysis/           # Output classification, metrics & reporting
│   ├── judge.py        # CoT-based semantic safety judge (local or API)
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

### Pipeline Flow

```
┌──────────┐    ┌─────────────┐    ┌───────────┐    ┌──────────┐
│  Attack   │───▶│ Consolidate │───▶│   Judge   │───▶│  Report  │
│           │    │             │    │           │    │          │
│ PromptMap │    │ JSON → CSV  │    │ CoT-based │    │ TXT/JSON │
│ PAIR      │    │ Standardize │    │ 0/1 score │    │ PDF+Chart│
│ CipherChat│    │ entries     │    │ reasoning │    │ Metrics  │
└──────────┘    └─────────────┘    └───────────┘    └──────────┘
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
target_model: gpt-4.1-2025-04-14
target_model_type: openai

judge_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
judge_model_type: local

attacks:
  - promptmap
  - pair
  - cipherchat

dataset: advbench
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

# Consolidate
consolidator = ResultsConsolidator()
df = consolidator.consolidate_from_json_files(["results/promptmap.json"])

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

A higher Robustness Score indicates a model that requires more complex attacks to compromise, suggesting stronger safety alignment.

---

## :brain: CoT Judge

The Chain-of-Thought judge is the core semantic evaluation component. Unlike keyword-based classifiers, it uses a reasoning-capable LLM to analyze the **intent** behind model responses.

### Supported Backends

| Backend | Config | VRAM | Cost |
|---------|--------|------|------|
| DeepSeek-R1-Distill-Qwen-14B (4-bit) | `--judge-model-type local` | ~9 GB | Free |
| DeepSeek-R1-Distill-Llama-8B | `--judge-model-type local` | ~16 GB | Free |
| GPT-4.1 | `--judge-model-type openai` | — | API cost |
| Any OpenAI-compatible model | `--judge-model-type openai` | — | API cost |

The local judge uses 4-bit NF4 quantization via `bitsandbytes` to fit large models in consumer GPU VRAM.

### Output

For each entry, the judge produces:
- `judge_result`: YES (jailbroken), NO (refused), or AMBIGUOUS
- `judge_justification`: brief explanation
- `judge_reasoning_chain`: full CoT reasoning trace

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