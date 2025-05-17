# nlp_interpretability
Final project for a Natural Language Processing course, focusing on the interpretability of Transformer-based neural networks for sentiment analysis.

## Project Goal
This project aims to analyze and understand the decision-making process of a Transformer model fine-tuned for sentiment classification on the IMDB dataset ("columbine/imdb-dataset-sentiment-analysis-in-csv-format"). We will leverage the SHAP (SHapley Additive exPlanations) library to generate feature attributions, identifying which words or tokens most significantly influence the model's predictions for positive or negative sentiment.

## Methodology
1.  **Dataset:** IMDB movie reviews dataset (CSV format).
2.  **Model:** A Transformer-based architecture (e.g., BERT, DistilBERT) from the Hugging Face library. The model will be fine-tuned for binary sentiment classification (positive/negative).
3.  **Interpretability Tool:** SHAP (SHapley Additive exPlanations).
4.  **Process:**
    *   Load and preprocess the dataset.
    *   Select and fine-tune a pre-trained Transformer model.
    *   Implement a SHAP explainer to analyze model predictions on test instances.
    *   Visualize and interpret SHAP values to understand key features driving sentiment.
5.  **Expected Outcome:** Insights into the model's behavior, identification of influential textual features, and a clearer understanding of how the Transformer model arrives at its sentiment classifications.

## Tech Stack
- Python
- Hugging Face `transformers` & `datasets`
- PyTorch (or TensorFlow)
- SHAP
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)
