```markdown
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

**Features used (14 total):**
- GPS position: latitude, longitude
- Power statistics: max, mean, std, median, top-3 mean, range
- Mobility: altitude, distance, speed, height, z-speed, pitch

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
| Gemini (few-shot) | LLM | General-purpose LLM with few-shot prompting, no training |
| Qwen + LoRA | Fine-tuned LLM | Instruction-tuned LLM with LoRA adapters on beam prediction data |
| BeamLLM | LLM + Reprogramming | Frozen LLM backbone with trainable input/output reprogramming layers and classification head |

---

## Key Results

### Beam Prediction (14 features)

| Model | Top-1 | Top-2 | Top-3 |
|-------|-------|-------|-------|
| KNN | 0.657 | 0.801 | 0.823 |
| MLP | 0.640 | 0.862 | 0.935 |
| **Random Forest** | **0.697** | 0.864 | 0.922 |
| Gemini (few-shot) | 0.100 | 0.210 | 0.250 |
| Qwen + LoRA | 0.340 | 0.430 | 0.540 |
| **BeamLLM** | **0.692** | **0.883** | **0.943** |

**Top-K Accuracy:** percentage of test samples where the correct beam appears within the top-K predictions.

### Ablation Study: 8 vs 14 Features (Top-1)

| Model | 8 features | 14 features | Δ |
|-------|-----------|------------|---|
| Random Forest | 0.625 | 0.697 | **+7.2%** |
| KNN | 0.596 | 0.657 | **+6.1%** |
| MLP | 0.649 | 0.640 | -0.9% |
| Gemini | 0.140 | 0.100 | -4.0% |
| Qwen + LoRA | 0.300 | 0.340 | **+4.0%** |
| BeamLLM | 0.638 | 0.692 | **+5.4%** |

### Handover Detection (θ=10, 14 features)

| Model | HDA | FAR | MDR |
|-------|-----|-----|-----|
| **Random Forest** | **0.970** | **0.035** | **0.023** |
| **KNN** | **0.970** | **0.035** | **0.023** |
| MLP | 0.940 | 0.053 | 0.070 |
| Gemini | 0.570 | 0.193 | 0.744 |
| Qwen + LoRA | 0.590 | 0.211 | 0.674 |
| BeamLLM | 0.940 | 0.053 | 0.070 |
| Ground Truth | 1.000 | 0.000 | 0.000 |

---

## Additional Contribution: Handover Detection

Beyond beam prediction, the project introduces a **handover detection module** that identifies when a drone should switch its communication link to a different beam configuration. Each model's predictions are analyzed as a time series, and handover events are detected when the beam index changes abruptly beyond a threshold θ. The module evaluates:

- **HDA** — Handover Detection Accuracy
- **FAR** — False Alarm Rate
- **MDR** — Miss Detection Rate
- **Sensitivity Analysis** — HDA across different thresholds (θ = 5, 10, 15, 20)

---

## Research Findings

1. **Mobility features significantly improve performance.** Adding 6 mobility features (altitude, distance, speed, height, z-speed, pitch) improved Random Forest by +7.2%, KNN by +6.1%, and BeamLLM by +5.4%. Tree-based models benefit most, as they effectively exploit the categorical patterns introduced by altitude and distance features. In handover detection, Random Forest and KNN improved from 91% and 90% to 97% HDA respectively.

2. **Zero-shot LLMs fail at domain-specific tasks.** Gemini achieves only 10% Top-1 accuracy with few-shot prompting, confirming that general language knowledge does not transfer to wireless communication tasks. Adding more features to the prompt actually hurt performance (from 14% to 10%), showing that more information without domain knowledge only adds confusion.

3. **Fine-tuning helps but is insufficient with text generation.** Qwen + LoRA improves to 34% Top-1 with 14 features (+4% over 8 features), showing that fine-tuning provides meaningful adaptation but text generation is not the right paradigm for numerical classification.

4. **Architectural reprogramming bridges the gap.** BeamLLM, using a frozen LLM backbone with trainable reprogramming layers, achieves 69.2% Top-1 — surpassing all classical ML models in Top-2 (88.3%) and Top-3 (94.3%), and matching Random Forest in Top-1. Only 0.73% of parameters are trainable.

5. **Traditional ML remains competitive.** Random Forest with Grid Search achieves 69.7% Top-1, becoming the best classical model when mobility features are included. Well-tuned classical models are still strong baselines for structured sensor data tasks.

6. **Domain-specific pretraining is the missing ingredient.** The performance gap between BeamLLM and Random Forest suggests that LLMs require domain-specific pretraining on wireless data to fully exploit their representational capacity — consistent with the findings of Cheng et al.

---

## Project Structure

```
uav_beam_project/
    scenario23_dev/                   DeepSense 6G Scenario 23 dataset
    LoadFile.py                       Load dataset and extract 14 features
    Preproccesing.py                  Normalization and train/val/test split
    ML_models.py                      Train RF, KNN, MLP with Grid Search
    gemini.py                         Gemini API few-shot evaluation
    handover.py                       Handover detection and metrics
    comparison.py                     Final plots and summary table
    FineTuningLLM_BeamPredection.ipynb   Qwen + LoRA fine-tuning (Google Colab)
    BeamLLM_BeamPredictor.ipynb          BeamLLM reprogramming architecture (Google Colab)
    requirements.txt                  Python dependencies
    results_backup_8features/         Backup of results with 8 features
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
python LoadFile.py           # Build clean dataset with 14 features
python Preproccesing.py      # Preprocess and split
python ML_models.py          # Train ML models with Grid Search
python gemini.py             # Run Gemini few-shot evaluation
# Run FineTuningLLM_BeamPredection.ipynb on Google Colab → download lora_results.pkl
# Run BeamLLM_BeamPredictor.ipynb on Google Colab → download beamllm_results.pkl
python handover.py           # Handover detection
python comparison.py         # Final comparison plots
```

---

## Requirements

- Python 3.10+
- scikit-learn, numpy, pandas, matplotlib
- google-genai
- torch, transformers, peft, trl (for Colab notebooks)
- Google Colab with T4 GPU (for fine-tuning notebooks)

---

## References

Lu Cheng, Hongliang Zhang, Boya Di, Dusit Niyato, Lingyang Song,
*"Large Language Models Empower Multimodal Integrated Sensing and Communication,"*
IEEE Communications Magazine, May 2025.

C. Zheng et al.,
*"BeamLLM: Vision-Empowered mmWave Beam Prediction with Large Language Models,"*
arXiv preprint arXiv:2503.10432, Mar. 2025.

A. Alkhateeb et al.,
*"DeepSense 6G: A Large-Scale Real-World Multi-Modal Sensing and Communication Dataset,"*
IEEE Communications Magazine, vol. 61, no. 9, Sep. 2023.
```