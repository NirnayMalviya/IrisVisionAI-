# IrisVision AI — Full MVP

A multi-page Streamlit app for Iris species classification with EDA, model
comparison across 6 algorithms, live predictions, and SHAP explainability —
styled with a grayscale liquid-glass morphism theme.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Pages

- **Home** — hero + dataset preview
- **Explore Data** — dataset table, CSV upload, summary stats, pairplot,
  correlation heatmap, box/violin plots, class distribution
- **Train & Compare Models** — trains KNN, Decision Tree, Logistic
  Regression, SVM, Random Forest, Naive Bayes; leaderboard; confusion
  matrix; classification report; ROC curves; feature importance; save
  best model with joblib
- **Predict** — sliders + quick-fill examples, live prediction with class
  probabilities, SHAP waterfall explanation + plain-language summary,
  downloadable session history
- **About** — plain-language methodology

## Project structure

```
app.py                          # Home page (entry point)
pages/
  1_Explore_Data.py
  2_Train_Compare_Models.py
  3_Predict.py
  4_About.py
utils/
  data_loader.py                # dataset loading + CSV validation
  preprocessing.py               # train/test split + scaling
  training.py                    # trains all 6 models
  evaluation.py                  # confusion matrix / ROC / feature importance
  explainability.py              # SHAP explanations
  persistence.py                 # save/load models with joblib
  styling.py                     # grayscale glassmorphism theme + CSS
models/                          # saved model artifacts land here
.streamlit/config.toml           # dark theme config
requirements.txt
```

## Deploying

1. Push this project to a GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub.
3. Point it at your repo + app.py, click deploy.
