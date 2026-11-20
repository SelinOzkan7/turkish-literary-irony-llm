# Lost in Literary Irony: Evaluating Cross-Lingual and Translation Effects on Turkish Literary Irony Comprehension in LLMs

**Course:** Natural Language Processing — Università degli Studi di Milano  
**Track:** Track 5 — Literature and Language: LLMs as Readers and as Writers  
**Reference Paper:** Khoshtab et al. (2025), *"Comparative Study of Multilingual Idioms and Similes in Large Language Models"*, Proceedings of COLING 2025 (pp. 8680–8698).

---

## Overview & Research Questions
While Khoshtab et al. (COLING 2025) benchmarked multilingual LLMs on short, isolated similes and idioms, they noted the lack of context-dependent figurative benchmarks covering **irony and sarcasm**. This repository introduces **TR-LitIrony**, a 60-passage parallel literary benchmark curated from Ahmet Hamdi Tanpınar's *Saatleri Ayarlama Enstitüsü* (*The Time Regulation Institute*) and Orhan Pamuk's *Benim Adım Kırmızı* (*My Name is Red*).

We investigate three core research questions across **960 controlled evaluations**:
1. **RQ1 (Original vs. Translated Input):** Do open-source multilingual LLMs (`Qwen2.5-7B-Instruct`, `Qwen2.5-3B-Instruct`) comprehend Turkish literary irony more accurately in original Turkish (`native_tr`) or in English translation?
2. **RQ2 (Human Literary Translation vs. Machine Translation):** Does published professional human translation (`human_en`) preserve pragmatic and ironic nuances better than neural machine translation (`mt_en`, Meta NLLB-200)?
3. **RQ3 (Reasoning & Cross-Lingual Prompting):** How do Chain-of-Thought (`cot`) and Cross-Lingual Thought (`native_and_en`) strategies affect susceptibility to literal misinterpretations?

---

## Key Experimental Results (N = 60 Passages)

| Model | Condition | Zero-Shot Acc | CoT Acc | Zero-Shot F1 | CoT F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B-Instruct** | `native_tr` (Original Turkish) | **91.67%** | 80.00% | 0.9138 | 0.7964 |
| **Qwen2.5-7B-Instruct** | `human_en` (Literary Human Translation) | **91.67%** | 83.33% | 0.9138 | 0.8316 |
| **Qwen2.5-7B-Instruct** | `mt_en` (Neural Machine Translation - NLLB-200) | 85.00% | 76.67% | 0.8479 | 0.7643 |
| **Qwen2.5-7B-Instruct** | `native_and_en` (Cross-Lingual Thought - XLT) | **95.00%** | **95.00%** | **0.9475** | **0.9475** |
| **Qwen2.5-3B-Instruct** | `native_tr` (Original Turkish) | **98.33%** | 88.33% | 0.9822 | 0.8732 |
| **Qwen2.5-3B-Instruct** | `human_en` (Literary Human Translation) | **98.33%** | 93.33% | 0.9819 | 0.9282 |
| **Qwen2.5-3B-Instruct** | `mt_en` (Neural Machine Translation - NLLB-200) | 96.67% | 88.33% | 0.9634 | 0.8756 |
| **Qwen2.5-3B-Instruct** | `native_and_en` (Cross-Lingual Thought - XLT) | 88.33% | 88.33% | 0.8793 | 0.8793 |

### Main Findings
* **Machine Translation Degradation:** `mt_en` consistently underperforms both `native_tr` and `human_en` across all models and prompting strategies (e.g., dropping from 91.67% to 85.00% on `Qwen2.5-7B`), showing that NLLB-200 flattens subtle ironic cues.
* **The Chain-of-Thought (CoT) Paradox:** Replicating the scaling trend observed by Khoshtab et al. (COLING 2025), CoT reasoning degrades accuracy by 5%–11.7% on 3B–7B models due to literal over-analysis of surface lexical cues.
* **Scale-Dependent XLT Effect:** Cross-lingual self-translation (`native_and_en`) boosts `Qwen2.5-7B` to its peak accuracy (95.00%) but degrades the smaller `Qwen2.5-3B` model (88.33%) due to cross-lingual cognitive interference.

---

## Repository Structure
```text
├── data/
│   └── tr_lit_irony.jsonl          # 60 balanced literary passages (20 verbal irony, 20 satire, 20 literal control)
├── src/
│   ├── data_loader.py              # Dataset loading & schema validation (IronyDatasetLoader)
│   ├── prompt_builder.py           # Zero-shot, CoT, and XLT prompt templates (PromptBuilder)
│   ├── model_runner.py             # 4-bit quantized HuggingFace inference wrapper (LLMRunner)
│   ├── evaluator.py                # Regex answer extraction, Accuracy & Macro-F1 metrics (IronyEvaluator)
│   ├── visualizer.py               # Publication-ready plots (ResultVisualizer)
│   └── translate_mt.py             # Automated TR->EN translation via Meta NLLB-200
├── notebooks/
│   └── experiment_demo.ipynb       # Demonstration & visualization notebook
├── results/
│   ├── detailed_predictions.csv    # Instance-level model outputs across all 960 runs
│   ├── summary_metrics.csv         # Aggregated Accuracy & Macro-F1 scores
│   └── figures/
│       └── condition_comparison.png
├── main.py                         # Reproducible CLI experiment runner
├── requirements.txt
├── README.md
└── report.pdf                      # Final research article (LaTeX, 4–8 pages)