from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# ======================================
# LOAD DATA
# ======================================

df = pd.read_csv("rainfall in india 1901-2015.csv")

df.columns = df.columns.str.strip().str.upper()
df = df.dropna()

location_col = "SUBDIVISION"

months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
month_cols = [m for m in months if m in df.columns]

monsoon_cols = ['JUN','JUL','AUG','SEP']
df['MONSOON_AVG'] = df[monsoon_cols].mean(axis=1)

df['ANNUAL'] = df[month_cols].sum(axis=1)

# Locations list
locations = sorted(df[location_col].unique())

# ======================================
# TRAIN MODEL (GLOBAL)
# ======================================

X = df[month_cols + ['MONSOON_AVG']]
y = df['ANNUAL']

selector = SelectKBest(score_func=f_regression, k=5)
X = X[X.columns[selector.fit(X,y).get_support()]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

avg_rainfall = df['ANNUAL'].mean()

# ======================================
# SEVERITY FUNCTION
# ======================================

def severity_rate(predicted):

    if predicted < 0.7 * avg_rainfall:
        return "High Drought Risk"

    elif predicted > 1.3 * avg_rainfall:
        return "High Flood Risk"

    else:
        return "Normal"


# ======================================
# WEB ROUTE
# ======================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    risk = None
    selected_location = None
    chart_data = None

    if request.method == "POST":

        selected_location = request.form["location"]
        selected_month = request.form["month"]

        # Filter location
        location_df = df[df[location_col] == selected_location]

        # Prediction
        avg_values = X.mean().values.reshape(1, -1)
        avg_scaled = scaler.transform(avg_values)

        predicted_annual = model.predict(avg_scaled)[0]

        month_ratio = location_df[selected_month].mean() / location_df['ANNUAL'].mean()

        prediction = round(predicted_annual * month_ratio, 2)

        risk = severity_rate(predicted_annual)

        # Chart data
        chart_data = location_df[month_cols].mean().tolist()

    return render_template(
        "index.html",
        locations=locations,
        months=month_cols,
        prediction=prediction,
        risk=risk,
        selected_location=selected_location,
        chart_data=chart_data
    )


# ======================================
# RUN APP
# ======================================

if __name__ == "__main__":
    app.run(debug=True)