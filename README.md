# SHAP-Based Interpretability Analysis for BERT Sentiment Classification

A comprehensive research project investigating the reliability of SHAP explanations for BERT-based sentiment classification models, with particular focus on how model overfitting affects explanation quality.

##  Research Overview

This project demonstrates a **critical finding in interpretability research**: **overfitted BERT models produce counterintuitive and unreliable SHAP explanations**, even when achieving high accuracy (93%). Our analysis reveals that explanation quality is not guaranteed by model performance alone.

### Key Findings
-  **Fine-tuned BERT achieved 93% accuracy** on IMDB sentiment classification
-  **SHAP explanations were highly unreliable** due to overfitting
-  **Counterintuitive attributions**: Words like "forgotten" attributed to positive sentiment
-  **Minimal scores for obvious sentiment words**: "poor" and "bad" scored near zero

##  Project Structure

```
nlp_interpretability/
├── notebook.ipynb                      # Main implementation notebook
├── INTERPRETABILITY_RESEARCH_REPORT.md # Comprehensive research report
├── SHAP_Interpretation_Guide.md        # Technical implementation guide
├── README.md                           # This file
└── requirements.txt                    # Dependencies (to be created)
```

##  Quick Start


### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nlp_interpretability.git
   cd nlp_interpretability
   ```

2. **Install dependencies**
   ```bash
   pip install torch transformers datasets shap pandas scikit-learn matplotlib numpy tqdm
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook notebook.ipynb
   ```

## 🔬 Methodology

### Dataset
- **Source**: IMDB Movie Review Dataset (50,000 reviews)
- **Task**: Binary sentiment classification (positive/negative)
- **Split**: 80/20 train/validation for overfitting detection
- **Preprocessing**: BERT tokenization with 256 max sequence length

### Models Tested
1. **Baseline BERT**: Minimal fine-tuning (87% train, 85% validation accuracy)
2. **Fine-tuned BERT**: Extensive fine-tuning (98.5% train, 93% validation accuracy)

### SHAP Analysis
- **Method**: Transformer-specific SHAP implementation
- **Scope**: Token-level attributions for individual predictions
- **Background**: 50 representative samples from training set
- **Evaluation**: Linguistic plausibility and consistency metrics

##  Results Summary

### Model Performance
| Model | Training Acc | Validation Acc | Overfitting Gap |
|-------|-------------|----------------|-----------------|
| Baseline BERT | 87% | 85% | 2% |
| Fine-tuned BERT | 98.5% | 93% | 5.5% |

### SHAP Analysis Examples

** Counterintuitive Results for Negative Classifications:**
- `"ridiculous"`: +0.007 (positive attribution for negative word)
- `"worst"`: -0.102 (minimal attribution for strongly negative word)
- `"enjoyable"`: +0.018 (positive attribution in negative classification)
- `"boring"`: -0.149 (expected but weak attribution)


### Video Link
[Video Link](https://drive.google.com/file/d/1-_0000000000000000000000000000000000000000/view?usp=sharing)
