# VisPruner Experiments

This repository provides baseline and modified **LLaVA-1.5** evaluation scripts on the **POPE** benchmark, including:

* A vanilla LLaVA baseline
* The original **VisPruner** method
* A VisPruner variant that explicitly **excludes attention sink tokens**

---

## 1. Environment Setup

* **Python version**: `3.12.12`

Install all required dependencies via:

```bash
pip install -r requirements.txt
```

---

## 2. Scripts Overview

### 2.1 `llava_baseline.py`

Runs the **standard LLaVA-1.5** model without any visual token pruning. This serves as the baseline for comparison.

**Usage:**

```bash
python llava_baseline.py \
    --pope_path lmms-lab/POPE \
    --pope_catagory adversarial \
    --pope_max_new_token 8 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --output_file ./llava_baseline.txt
```

---

### 2.2 `llava_vispruner.py`

Implements the **original VisPruner** method, which prunes visual tokens based on attention statistics.

**Key arguments:**

* `--retain_img_tokens`: Number of image tokens kept after pruning
* `--vispruner_ratio`: Ratio used to select important tokens
* `--removal_number`: Maximum number of tokens removed per image

**Usage:**

```bash
python llava_vispruner.py \
    --pope_path lmms-lab/POPE \
    --pope_catagory adversarial \
    --pope_max_new_token 8 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --retain_img_tokens 32 \
    --vispruner_ratio 0.1 \
    --removal_number 64 \
    --output_file ./llava_vispruner.txt
```

---

### 2.3 `llava_vispruner_exclude_attention_sink.py`

Extends VisPruner by **explicitly filtering out attention sink tokens** before pruning. Attention sinks are identified using a threshold on attention mass.

**Additional argument:**

* `--attention_sink_threshold`: Tokens with attention scores above this threshold are treated as sinks and excluded

**Usage:**

```bash
python llava_vispruner_exclude_attention_sink.py \
    --pope_path lmms-lab/POPE \
    --pope_catagory adversarial \
    --pope_max_new_token 8 \
    --model_path llava-hf/llava-1.5-7b-hf \
    --retain_img_tokens 32 \
    --vispruner_ratio 0.1 \
    --removal_number 64 \
    --attention_sink_threshold 0.02 \
    --output_file ./llava_vispruner_exclude_attention_sink.txt
```

---

## 3. Notes

* All experiments are evaluated on the **POPE adversarial split**.
* The maximum number of generated tokens is fixed to ensure fair comparison.
* Output files store the raw model predictions for downstream analysis.

---

## 4. Intended Use

This codebase is intended for:

* Studying **attention sinks** in vision-language models
* Analyzing the robustness of **visual token pruning**
* Comparing baseline vs. pruned VLM behavior under hallucination-sensitive benchmarks

---

If you modify pruning strategies or attention filtering rules, we recommend keeping the evaluation setup identical for reproducibility.
