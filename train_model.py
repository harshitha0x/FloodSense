import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

flood = pd.read_csv("flood_history.csv")
flood = flood.dropna(subset=["DATE", "FLOOD"])
res = pd.read_csv("chennai_reservoir_levels.csv")

flood["DATE"] = pd.to_datetime(flood["DATE"], dayfirst=True)
res["Date"] = pd.to_datetime(res["Date"], dayfirst=True)

res["RES_TOTAL"] = res[["POONDI", "CHOLAVARAM", "REDHILLS", "CHEMBARAMBAKKAM"]].sum(axis=1)

df = pd.merge(flood, res, left_on="DATE", right_on="Date", how="left")
df = df.sort_values("DATE")

df["RAIN_3DAY"] = df["RAINFALL"].rolling(3, min_periods=1).sum()
df["RAIN_5DAY"] = df["RAINFALL"].rolling(5, min_periods=1).sum()
df["RAIN_7DAY"] = df["RAINFALL"].rolling(7, min_periods=1).sum()
df["RES_TOTAL"] = df["RES_TOTAL"].ffill().fillna(0)

# cap extremes so a few huge days do not dominate
for col in ["RAINFALL", "RAIN_3DAY", "RAIN_5DAY", "RAIN_7DAY"]:
    df[col] = df[col].clip(upper=df[col].quantile(0.99))

X = df[["RAINFALL", "RAIN_3DAY", "RAIN_5DAY", "RAIN_7DAY", "RES_TOTAL"]]
y = df["FLOOD"]  # nearby days already labelled 1 in flood_history.csv

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
)
model.fit(X, y)
joblib.dump(model, "flood_model_from_script.pkl")
print("saved flood_model_from_script.pkl", "rows:", len(df))