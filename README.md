# UAV Beam Prediction and Handover Detection
## Using Multimodal Sensing and Large Language Models

---

## Overview

This project investigates the use of **Large Language Models (LLMs)** for beam prediction and handover detection in drone-to-infrastructure (UAV) wireless communication systems. It is inspired by the paper *"Large Language Models Empower Multimodal Integrated Sensing and Communication"* (Cheng et al., IEEE Communications Magazine, May 2025), which proposes the integration of MLLMs into Integrated Sensing and Communication (ISAC) systems for 6G networks.

The core research question is:

> **Can Large Language Models compete with traditional Machine Learning models in domain-specific wireless communication tasks such as beam prediction?**

---

## Motivation

In modern mmWave communication systems, selecting the optimal beam from a large codebook is critical for maintaining link quality — especially for fast-moving platforms like drones. Traditional approaches rely on ML classifiers trained on sensor data. This project explores whether LLMs, through different levels of adaptation (zero-shot, fine-tuning, and architectural reprogramming), can match or surpass these baselines.

---

## Dataset

**DeepSense 6G — Scenario 23 (Drone-to-Infrastructure)**
- Source: [https://www.deepsense6g.net](https://www.deepsense6g.net)
- 11,387 real-world measurement samples
- A drone flies over a flying zone in front of a fixed 60 GHz base station
- The base station sweeps all 64 beams and records received power per beam

**Features used (8 total):**
- GPS position: latitude, longitude
- Power statistics: max, mean, std, median, top-3 mean, range

**Labels:**
- `unit1_beam_index`: optimal beam (argmax of power vector)
- `unit1_beam_top2`, `unit1_beam_top3`: second and third best beams

**Split:** 70% train / 10% validation / 20% test

---

## Models Compared

| Model | Type | Description |
|-------|------|-------------|
| Random Forest | Traditional ML | Ensemble of decision trees with Grid Search |
| KNN | Traditional ML | K-Nearest Neighbors with Grid Search |
| MLP | Traditional ML | Multilayer Perceptron with Grid Search |
| Gemini (zero-shot) | LLM | General-purpose LLM with few-shot prompting, no training |
| Qwen + LoRA | Fine-tuned LLM | Instruction-tuned LLM with LoRA adapters on beam prediction data |
| BeamLLM | LLM + Reprogramming | Frozen LLM backbone with trainable input/output reprogramming layers and classification head |

---

## Key Results

| Model | Top-1 | Top-2 | Top-3 |
|-------|-------|-------|-------|
| KNN | 0.596 | 0.767 | 0.820 |
| Random Forest | 0.625 | 0.810 | 0.880 |
| BeamLLM | 0.638 | 0.832 | 0.907 |
| MLP | 0.649 | 0.837 | 0.911 |
| Gemini (zero-shot) | 0.010 | 0.070 | 0.100 |
| Qwen + LoRA | 0.140 | 0.190 | 0.200 |

**Top-K Accuracy:** percentage of test samples where the correct beam appears within the top-K predictions.

---

## Additional Contribution: Handover Detection

Beyond beam prediction, the project introduces a **handover detection module** that identifies when a drone should switch its communication link to a different beam configuration. Each model's predictions are analyzed as a time series, and handover events are detected when the beam index changes abruptly beyond a threshold θ. The module evaluates:

- **HDA** — Handover Detection Accuracy
- **FAR** — False Alarm Rate
- **MDR** — Miss Detection Rate
- **Sensitivity Analysis** — HDA across different thresholds (θ = 5, 10, 15, 20)

---

## Research Findings

1. **Zero-shot LLMs fail at domain-specific tasks.** Gemini achieves only 1% Top-1 accuracy without any domain adaptation, confirming that general language knowledge does not transfer to wireless communication tasks.

2. **Fine-tuning helps but is insufficient with text generation.** Qwen + LoRA improves from 1% to 14% Top-1, showing that fine-tuning provides meaningful adaptation but text generation is not the right paradigm for numerical classification.

3. **Architectural reprogramming bridges the gap.** BeamLLM, using frozen LLM backbone with trainable reprogramming layers, achieves 63.8% Top-1 — surpassing Random Forest and KNN, and approaching MLP performance within 1.1%.

4. **Traditional ML remains competitive.** MLP with Grid Search achieves the best overall performance, highlighting that well-tuned classical models are still strong baselines for structured sensor data tasks.

5. **Domain-specific pretraining is the missing ingredient.** The performance gap between BeamLLM and MLP suggests that LLMs require domain-specific pretraining on wireless data to fully exploit their representational capacity — consistent with the findings of Cheng et al.

---

## Project Structure

```
uav_beam_project/
    scenario23_dev/              DeepSense 6G Scenario 23 dataset
    step1_load_data.py           Load dataset and extract features
    step2_preprocessing.py       Normalization and train/val/test split
    step3_ml_models.py           Train RF, KNN, MLP with Grid Search
    step4_gemini.py              Gemini API zero-shot evaluation
    step5_handover.py            Handover detection and metrics
    step6_final_comparison.py    Final plots and summary table
    lora_finetuning.ipynb        Qwen + LoRA fine-tuning (Google Colab)
    beamllm_reprogramming.ipynb  BeamLLM reprogramming architecture (Google Colab)
    requirements.txt             Python dependencies
```

---

## Installation

```bash
pip install -r requirements.txt
```

For the Gemini API:
```bash
pip install google-genai
```

Get a free API key at: [https://aistudio.google.com](https://aistudio.google.com)

---

## Execution Order

```bash
python step1_load_data.py        # Build clean dataset
python step2_preprocessing.py    # Preprocess and split
python step3_ml_models.py        # Train ML models
python step4_gemini.py           # Run Gemini evaluation
# Run lora_finetuning.ipynb and beamllm_reprogramming.ipynb on Google Colab
python step5_handover.py         # Handover detection
python step6_final_comparison.py # Final comparison plots
```

---

## Requirements

- Python 3.10+
- scikit-learn, numpy, pandas, matplotlib
- google-genai
- torch, transformers, peft, trl (for Colab notebooks)
- Google Colab with T4 GPU (for fine-tuning notebooks)

---

## Reference

Lu Cheng, Hongliang Zhang, Boya Di, Dusit Niyato, Lingyang Song,
*"Large Language Models Empower Multimodal Integrated Sensing and Communication,"*
IEEE Communications Magazine, May 2025.

Dataset: A. Alkhateeb et al., *"DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,"* IEEE Communications Magazine, vol. 61, no. 9, Sep. 2023.