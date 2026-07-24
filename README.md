# 🌱 CarbonAI — AI-Powered Sustainability Platform

CarbonAI is a Streamlit web app that combines **Machine Learning** and **Deep Learning** to help users understand and reduce their environmental impact:

1. **🌍 Carbon Footprint Calculator** — predicts a user's estimated monthly carbon emissions (kg CO₂) from lifestyle, transport, and energy-use inputs using a trained regression model.
2. **♻️ AI Waste Classification** — classifies an uploaded photo of waste into one of six categories (cardboard, glass, metal, organic, paper, plastic) using a fine-tuned MobileNetV2 CNN, and collects user feedback to track model performance.

---

## ✨ Features

- Clean, dark-themed Streamlit UI with a home page, calculator, waste classifier, and an admin feedback dashboard
- Carbon emission prediction with color-coded bands (🟢 Low / 🟡 Moderate / 🔴 High)
- Image-based waste classification with confidence scores
- User feedback loop (👍/👎) logged to `feedback.csv` for monitoring prediction accuracy
- End-to-end training pipeline: raw data → preprocessing → feature engineering → model training

---

## 🗂 Project Structure

```
.
├── app.py                     # Streamlit application (main entry point)
├── preprocessing.py           # Cleans the raw carbon emissions dataset
├── feature_engineering.py     # Encodes categorical features, saves encoders
├── train_model.py             # Trains & compares regression models, saves the best one
├── carbon_footprint.ipynb     # Exploratory analysis / experimentation notebook
├── requirements.txt           # Python dependencies
├── model/
│   ├── carbon_model.pkl       # Trained regression model (Gradient Boosting, tuned)
│   ├── encoders.pkl           # Fitted LabelEncoders for categorical fields
│   └── mobilenet_finetuned_best.keras  # Fine-tuned MobileNetV2 waste classifier
├── data/
│   ├── Carbon Emission.csv    # Raw dataset (not tracked in git)
│   ├── CleanedDataset.csv     # Output of preprocessing.py
│   └── ProcessedDataset.csv   # Output of feature_engineering.py
├── feedback.csv               # User feedback log (not tracked in git)
└── .gitignore
```

> **Note:** Models, datasets, and feedback logs are excluded from version control via `.gitignore`. See [Setup](#-setup) below for how to obtain/generate them.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| App/UI | Streamlit |
| ML | scikit-learn, XGBoost |
| DL | TensorFlow / Keras (MobileNetV2) |
| Data | Pandas, NumPy |
| Utilities | Joblib |

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add the required assets
Since models and data are gitignored, place these in the project root before running the app:
- `model/carbon_model.pkl`
- `model/encoders.pkl`
- `model/mobilenet_finetuned_best.keras`

If you don't have them yet, you can regenerate the carbon-emission model from raw data (see [Training the Models](#-training-the-models)). The waste classification model (MobileNetV2) is trained separately and expected to already exist as a `.keras` file.

### 4. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🧪 Training the Models

### Carbon footprint regression model
1. Place the raw dataset at `data/Carbon Emission.csv`.
2. Clean it:
   ```bash
   python preprocessing.py
   ```
3. Encode categorical features:
   ```bash
   python feature_engineering.py
   ```
4. Train and select the best model (compares Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost, then tunes Gradient Boosting via grid search):
   ```bash
   python train_model.py
   ```
   This saves the final model to `model/carbon_model.pkl`.

### Waste classification model
Dataset: sourced from Kaggle.

The classifier was built in three stages (see `classifier.ipynb` for the full experimentation and training workflow):
1. **Baseline CNN** — a simple convolutional network trained from scratch to establish a performance baseline.
2. **Transfer learning (frozen MobileNetV2)** — MobileNetV2 pretrained on ImageNet used as a frozen feature extractor, with a custom classification head trained on top.
3. **Fine-tuning** — top layers of MobileNetV2 unfrozen and trained at a low learning rate to adapt the pretrained features to the waste dataset, producing the final model saved as `mobilenet_finetuned_best.keras`.
---

## 📊 Input Features (Carbon Footprint Calculator)

| Feature | Type |
|---|---|
| Transport | Categorical |
| Vehicle Type | Categorical |
| Vehicle Monthly Distance (km) | Numeric |
| Energy Efficiency | Categorical |
| Waste Bags per Week | Numeric |
| Recycling | Categorical (multi-select) |
| Monthly Grocery Bill | Numeric |
| Frequency of Air Travel | Categorical |

**Output:** Estimated monthly carbon emission (kg CO₂), bucketed as:
- 🟢 Low — under 1,500 kg CO₂
- 🟡 Moderate — 1,500–2,500 kg CO₂
- 🔴 High — above 2,500 kg CO₂

---

## ♻️ Waste Classification Classes

`cardboard`, `glass`, `metal`, `organic`, `paper`, `plastic`

Predictions include a confidence score, and users can submit feedback (correct/incorrect) which is logged to `feedback.csv` and reviewable on the **Feedback (Admin)** page.

---

## 📌 Roadmap Ideas

- Retrain the waste classifier periodically using collected feedback
- Add authentication to the admin feedback page
- Expose the carbon prediction model via an API endpoint
- Add historical tracking/trends for individual users

---

## 📄 License

Add your preferred license here (e.g., MIT).
