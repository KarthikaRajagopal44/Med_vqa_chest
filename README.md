# Medical VQA — Chest Baseline

A multimodal Visual Question Answering (VQA) system designed for chest X-rays. Given a radiological image and a clinical question, this model predicts a binary (yes/no) answer.

## 🛠 Tech Stack
* **Deep Learning Framework:** PyTorch
* **NLP & Datasets:** HuggingFace Datasets, DistilBERT
* **Experiment Tracking:** Weights & Biases (W&B)
* **Interpretability:** Grad-CAM
* **Metrics & Evaluation:** scikit-learn

## 🏗 Architecture & Methodology

### Dataset & Preprocessing
* **Source:** `flaviagiammarino/vqa-rad` dataset (2,248 QA pairs).
* **Filtration:** Filtered specifically for binary (yes/no) question-answer pairs, resulting in near-balanced splits (940 train / 251 test).
* **Risk Discovery:** Conducted rigorous Exploratory Data Analysis (EDA), including hash-based duplicate detection. Proactively identified 202 shared image hashes across train/test splits, flagging potential data leakage prior to modeling.

### Multimodal Model
* **Vision Branch:** CNN-based image feature extraction.
* **Text Branch:** Frozen DistilBERT embeddings (768-dimensional).
* **Fusion Strategy:** Modalities are fused via concatenation and passed through a Multi-Layer Perceptron (MLP) classification head.

### Evaluation Suite
Implemented a comprehensive evaluation pipeline to thoroughly assess model behavior:
* Accuracy, AUC-ROC, and Precision-Recall (PR) curves.
* Confusion matrices and confidence distribution analysis.
* Error slicing and visual interpretability using Grad-CAM.

## 📊 Results & Insights

| Metric | Score |
| :--- | :--- |
| **Train Accuracy** (Epoch 10) | 85.30% |
| **Test Accuracy** | 60.16% |
| **AUC-ROC** | 0.6465 |
| **Average Precision** | 0.6042 |

**Key Insight:** The generalization gap between train and test accuracy clearly reflects the image-level leakage risk. Because this issue was proactively flagged during the EDA phase, this baseline serves as an accurate, known-constraint benchmark rather than an unexpected failure.

## 🚀 Next Steps & Future Work
* Implement an **image-disjoint split strategy** to resolve the identified data leakage.
* Build config-driven, reproducible ablation pipelines.
* Explore stronger multimodal architectures (e.g., cross-attention mechanisms).
* Conduct model calibration experiments.
