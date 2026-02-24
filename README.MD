# Flood & Drought Predictor

A small Flask web app that predicts monthly rainfall and a simple flood/drought risk level for locations in India using a Random Forest regression model trained on historical rainfall data.

**Repository files**
- [app.py](app.py): main Flask application and model training/prediction code.
- [rainfall in india 1901-2015.csv](rainfall in india 1901-2015.csv): dataset (place in project root).
- [templates/index.html](templates/index.html): web UI template.

**Dataset source**
This project uses the Kaggle dataset "Rainfall in India 1901-2015" by aravindpcoder available at:

https://www.kaggle.com/datasets/aravindpcoder/rainfall-in-india-1901-2015

We downloaded the CSV from that Kaggle dataset and saved it here as `rainfall in india 1901-2015.csv`.

How the app works
- The CSV is loaded and cleaned in `app.py`.
- Monthly columns (JAN–DEC) are used to compute `MONSOON_AVG` (JUN–SEP) and `ANNUAL` totals.
- A feature selector (`SelectKBest`) keeps the top 5 features; features are scaled with `StandardScaler`.
- A `RandomForestRegressor` is trained to predict annual rainfall, then monthly predictions are derived by applying the location's historical month-to-annual ratio.
- Risk levels are computed from the predicted annual rainfall relative to the dataset mean:
  - < 0.7 × mean: High Drought Risk
  - > 1.3 × mean: High Flood Risk
  - otherwise: Normal

Requirements
- Python 3.8+
- Recommended packages (install via pip):

```
pip install flask pandas numpy scikit-learn
```

Or create `requirements.txt` with:

```
Flask
pandas
numpy
scikit-learn
```

Run locally
1. Ensure `rainfall in india 1901-2015.csv` is in the project root.
2. Install dependencies.
3. Start the app:

```
python app.py
```

4. Open http://127.0.0.1:5000/ in your browser.

Notes & assumptions
- The app trains a model at startup (quick but not optimized for production).
- The CSV filename must match exactly as used in `app.py`.
- `templates/index.html` contains a simple Chart.js chart for monthly averages.

If you want, I can:
- add a `requirements.txt` and `.gitignore`;
- move model training to a separate script and save a trained model to disk;
- or containerize the app with a Dockerfile.

---
Created/updated README for this project.
