import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("/mnt/user-data/uploads/weatherHistory.csv")

# --- Cleaning ---
df = df.dropna(subset=['Precip Type'])
df = df.drop_duplicates()
df = df[df['Pressure (millibars)'] != 0]
df = df[df['Humidity'] != 0]

# --- Proxy Cloudburst Risk Label ---
# Based on literature: cloudburst-prone conditions correlate with high humidity,
# low atmospheric pressure, reduced visibility, and elevated wind speed.
# Thresholds set using the 85th/15th percentiles of this dataset's own distribution
# (i.e. "top ~15% most extreme/unstable conditions" flagged as high risk).

humidity_thresh = df['Humidity'].quantile(0.85)       # ~0.93
pressure_thresh = df['Pressure (millibars)'].quantile(0.15)  # low pressure
visibility_thresh = df['Visibility (km)'].quantile(0.15)     # low visibility
windspeed_thresh = df['Wind Speed (km/h)'].quantile(0.85)    # high wind

print(f"Humidity threshold (>): {humidity_thresh:.3f}")
print(f"Pressure threshold (<): {pressure_thresh:.2f}")
print(f"Visibility threshold (<): {visibility_thresh:.2f}")
print(f"Wind speed threshold (>): {windspeed_thresh:.2f}")

# Flag as high risk if AT LEAST 3 of 4 unstable-weather conditions are met
conditions = (
    (df['Humidity'] > humidity_thresh).astype(int) +
    (df['Pressure (millibars)'] < pressure_thresh).astype(int) +
    (df['Visibility (km)'] < visibility_thresh).astype(int) +
    (df['Wind Speed (km/h)'] > windspeed_thresh).astype(int)
)
df['CloudburstRisk'] = (conditions >= 3).astype(int)

print()
print("Label distribution:")
print(df['CloudburstRisk'].value_counts())
print(f"High-risk %: {df['CloudburstRisk'].mean()*100:.2f}%")

x = df[["Temperature (C)","Apparent Temperature (C)","Humidity","Wind Speed (km/h)",
        "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)"]]
y = df["CloudburstRisk"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(x_train, y_train)
pred = model.predict(x_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)

print()
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

with open("/home/claude/cbmodel_cloudburst.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nSaved model to cbmodel_cloudburst.pkl")
