# MoodMeal Streamlit Analytics App

A flat, GitHub-ready Streamlit project for the MoodMeal synthetic survey dataset.

## Files in this repo
- `app.py` - Streamlit app
- `requirements.txt` - deployment dependencies
- `README.md` - setup notes
- `moodmeal_survey_synthetic_2200.csv` - raw survey-style dataset
- `moodmeal_model_ready_2200.csv` - encoded model-ready dataset
- `moodmeal_data_dictionary.csv` - column descriptions

## Deploy on Streamlit Community Cloud
1. Upload all files to the root of a GitHub repository.
2. Create a new app on Streamlit Community Cloud.
3. Point the app to `app.py`.
4. Deploy.

## Included analytics
- Classification with Accuracy, Precision, Recall, F1-Score, ROC-AUC, ROC Curve, Confusion Matrix, Feature Importance
- Clustering with K-Means, PCA projection, and silhouette score
- Association Rule Mining with Support, Confidence, and Lift
- Regression for average order value prediction with R², MAE, RMSE, and feature importance

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```
