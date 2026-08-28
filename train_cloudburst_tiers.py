import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("/mnt/user-data/uploads/weatherHistory.csv")

# --- Cleaning ---
df = df.dropna(subset=['Precip Type'])
df = df.drop_duplicates()
df = df[df['Pressure (millibars)'] != 0]
df = df[df['Humidity'] != 0]

# --- Thresholds (same as before) ---
humidity_thresh = df['Humidity'].quantile(0.85)
pressure_thresh = df['Pressure (millibars)'].quantile(0.15)
visibility_thresh = df['Visibility (km)'].quantile(0.15)
windspeed_thresh = df['Wind Speed (km/h)'].quantile(0.85)

conditions_met = (
    (df['Humidity'] > humidity_thresh).astype(int) +
    (df['Pressure (millibars)'] < pressure_thresh).astype(int) +
    (df['Visibility (km)'] < visibility_thresh).astype(int) +
    (df['Wind Speed (km/h)'] > windspeed_thresh).astype(int)
)

# --- 3-Tier Risk Label ---
# 0-1 conditions met -> Low (0)
# 2 conditions met    -> Medium (1)
# 3-4 conditions met  -> High (2)
def tier(c):
    if c <= 1:
        return 0  # Low
    elif c == 2:
        return 1  # Medium
    else:
        return 2  # High

df['RiskLevel'] = conditions_met.apply(tier)

print("Risk Level distribution:")
print(df['RiskLevel'].value_counts().sort_index())
print(f"\nLow: {(df['RiskLevel']==0).mean()*100:.2f}%  Medium: {(df['RiskLevel']==1).mean()*100:.2f}%  High: {(df['RiskLevel']==2).mean()*100:.2f}%")

x = df[["Temperature (C)","Apparent Temperature (C)","Humidity","Wind Speed (km/h)",
        "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)"]]
y = df["RiskLevel"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42, n_estimators=200)
model.fit(x_train, y_train)
pred = model.predict(x_test)

acc = accuracy_score(y_test, pred)
print(f"\nAccuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, pred, target_names=['Low','Medium','High']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# Feature importance
importances = model.feature_importances_
feat_names = x.columns
print("\nFeature Importances:")
for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    print(f"  {name:30s} {imp:.4f}")

with open("/home/claude/cbmodel_tiers.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nSaved model to cbmodel_tiers.pkl")
