
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
  <a href="#checkered_flag-usage">Usage</a> &#xa0; | &#xa0;
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
✔ Support for multiple LLM providers and deployment configurations  
✔ CoT-based semantic safety classification  
✔ Quantitative metrics (Attack Success Rate, Robustness Score)  
✔ Modular, extensible attack architecture  
✔ CLI and programmatic API  
✔ Alignment with the **NIST Adversarial Machine Learning taxonomy**

---

## :rocket: Architecture

SafeProbe is organized as a modular pipeline:

```

safeprobe/
├── attacks/        # Query-access attack implementations
│   ├── promptmap/  # Rule-based prompt transformation attacks
│   ├── pair/       # PAIR iterative refinement attack
│   └── cipherchat/ # Encoding-based attacks
│
├── datasets/       # Adversarial prompt datasets & model adapters
│
├── analysis/       # Output classification, metrics & reporting
│   ├── judge.py    # CoT-based semantic safety judge
│   ├── consolidator.py
│   └── report_gen.py
│
├── utils/          # Logging, API helpers, shared utilities
│
├── cli.py          # Command-line interface
└── config.py       # Configuration & pipeline orchestration

````

Each attack module can be executed independently or combined into a full evaluation run.  
New attacks can be added by implementing the provided module interface.

---

## :white_check_mark: Requirements

Before starting, ensure you have:

- **Python ≥ 3.10**
- **Git**
- API access to at least one supported LLM provider (e.g., OpenAI)

---

## :checkered_flag: Installation

Install directly from PyPI (or editable mode for development):

```bash
pip install safeprobe
````

Or from source:

```bash
git clone https://github.com/tompagli/safeprobe.git
cd safeprobe
pip install -e .
```

---

## :checkered_flag: Usage

### Command Line Interface

```bash
safeprobe --config config.yaml
```

### Programmatic Usage

```python
from safeprobe import config
from safeprobe.cli import main

main()
```

SafeProbe supports YAML-based configuration for:

* model backends
* attack selection
* dataset choice
* evaluation parameters
* output formats

---

## :bar_chart: Metrics

SafeProbe computes:

* **Attack Success Rate (ASR)**
  Proportion of adversarial prompts that successfully induce harmful outputs.

* **Robustness Score**
  A weighted score reflecting the complexity required to bypass model safety
  (simple prompts < role-play < multi-turn manipulation).

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


