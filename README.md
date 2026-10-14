# CloudBurst-Predictor

# Project Name: CloudBurst Prediction System
# PS Code: PRJ_238

## Team Name: [Add team name]
- Batch Number: 238
- Team Members:
  - Chandu Sai Darshan M (20231CSG0056)
  - V Govardhan Reddy (20231CSG0010)
  - Sethu Madhav Reddy (20231CSG0085)
- Supervisor: Mr. Asif Ahmed Najar, Assistant Professor, School of Computer Science and Engineering, Presidency University


## Project Overview 🚀
CloudBurst is a machine learning-based weather risk prediction system that classifies cloudburst-prone atmospheric conditions from standard meteorological readings. Because publicly available weather datasets rarely contain the rainfall-intensity ground truth needed to label real cloudburst events, this project addresses the gap through two complementary approaches:

1. A **proxy risk label** engineered from four meteorologically-supported correlates of atmospheric instability (humidity, pressure, visibility, wind speed), evaluated against seven classification algorithms on a large (94,602-row) public hourly weather dataset.
2. An independently constructed **real-observation validation dataset**, built by decoding raw hourly station reports from the NOAA Integrated Surface Database (ISD) across India's Himalayan cloudburst-prone belt, used to test whether the model's performance holds up against genuinely measured rainfall data rather than an engineered label alone.

Random Forest was selected as the final model based on its consistent performance across both evaluations, and is deployed as a Streamlit web application that automatically detects the user's location, fetches live weather conditions, and classifies cloudburst risk (High Risk / No Risk) in real time, with no manual data entry.

## Tech Stack 💻
- **Language:** Python 3
- **Machine Learning:** scikit-learn (Random Forest Classifier), Pandas, NumPy
- **Web Interface:** Streamlit
- **Data Source (primary):** Public hourly weather dataset (~96,000 records)
- **Data Source (validation):** NOAA Integrated Surface Database (ISD) — real hourly station reports, India Himalayan belt
- **Version Control:** Git/GitHub

## Project Structure 📁
```
CloudBurst-Predictor/
├── CloudBurst.py                  # Live-data Streamlit app (auto location + live weather)
├── CloudBurst_manual_backup.py    # Manual-entry version of the app (fallback)
├── cbmodel.pkl                    # Trained Random Forest model (proxy-labeled dataset)
├── requirements.txt
├── README.md
└── real_data_validation/          # Supplementary validation study (not used by the live app)
    ├── indian_weather_real.csv    # Decoded real NOAA station data (Dehradun, Shimla, Mukteshwar Kumaon)
    ├── build_real_dataset.py      # Pulls & decodes raw NOAA ISD station reports
    └── relabel_and_compare_v2.py  # Retrains & compares all 4 models on real observed data
```

> **Note:** `real_data_validation/` contains an independent research validation study — it confirms Random Forest remains the strongest model when tested against real measured rainfall data, not just the engineered proxy label. These files are **not** loaded by the deployed app; `CloudBurst.py` only uses `cbmodel.pkl`, trained on the primary proxy-labeled dataset.

## Project Features and Functionality ✨

- **Live Risk Classification:** Automatically approximates the user's location from their IP address (ip-api.com), retrieves current temperature, apparent temperature, humidity, wind speed, wind bearing, visibility and pressure from Open-Meteo, and predicts High Risk / No Risk.
- **Manual Fallback:** `CloudBurst_manual_backup.py` lets the user enter the seven readings by hand.
- **Interactive Web Interface:** Built with Streamlit for real-time, accessible predictions.
- **Validated Methodology:** Model choice supported by both a large-scale proxy-labeled evaluation and an independent real-observation validation study using genuine NOAA station data.

## How It Works 🛠️

```
Weather Dataset (CSV)
        |
        v
Data Preprocessing (cleaning + proxy risk-label construction)
        |
        v
Model Training (Random Forest, selected after comparing 7 classifiers)
        |
        v
Trained Model (cbmodel.pkl)
        |
        v
Streamlit Web App (IP-based location -> live weather from Open-Meteo -> unit conversion)
        |
        v
Risk Prediction (High Risk / No Risk)
```

## Methodology Summary 🧪

Since no publicly available dataset contains real-time rainfall-intensity ground truth at the scale needed for reliable model comparison, a proxy `CloudburstRisk` label was engineered from four percentile-thresholded indicators of atmospheric instability (humidity > 0.93 (85th percentile), pressure < 15th percentile, visibility < 15th percentile, wind speed > 85th percentile), flagging a reading as high-risk when at least 3 of 4 conditions co-occurred.

To validate this approach against genuine measured data, a second dataset was independently built by decoding raw NOAA ISD hourly station reports (Dehradun, Shimla, Mukteshwar Kumaon; 2005–2024), deriving an extreme-rainfall label directly from observed precipitation depth. Random Forest was the strongest or joint-strongest performer on both datasets, confirming the model choice generalizes beyond the engineered proxy label. See `real_data_validation/` for the full methodology and code.

## Limitations and Future Work 🚧

1. **Geographic scope:** The primary training dataset originates from a general public weather dataset, not India-specific data; the proxy label captures generalizable atmospheric-instability patterns but was not tuned to regional climatology.
2. **Temporal resolution:** NOAA ISD stations report precipitation at ~3-hour minimum granularity, likely under-detecting brief sub-hour rainfall bursts that define a true cloudburst.
3. **Sequence-aware modeling:** Future work could explore LSTM/GRU-based architectures to model the temporal build-up preceding extreme rainfall events, enabling forecasting ahead of an event rather than real-time classification.
4. **Live-app limitations:** IP-based geolocation is approximate (city-level); the app classifies current conditions rather than forecasting ahead; internet access is required.
5. **Expanded station coverage:** Incorporating more Himalayan-belt stations and satellite-derived precipitation products (e.g., NASA GPM IMERG) would strengthen real-observation validation further.

## How to Run ▶️
```
pip install -r requirements.txt
streamlit run CloudBurst.py
```
Run from the folder containing `cbmodel.pkl`.

## Screenshots and Demos 📸
<img width="1502" height="898" alt="Screenshot 2026-10-14 082958" src="https://github.com/user-attachments/assets/6cd21f99-fb7c-4bbb-b2e2-f61996d9c8a5" />



## Get In Touch! 📬
[Add team contact details here]

---

