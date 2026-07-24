import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("data/CleanedDataset.csv")
print(df.head())
import joblib
import os
categorical=[
"Transport",
    "Vehicle Type",
    "Energy efficiency",
    "Recycling",
    "Frequency of Traveling by Air"
]
encoders={}
for c in categorical:
    encoder=LabelEncoder()
    df[c]=encoder.fit_transform(df[c])
    encoders[c] = encoder
os.makedirs("model", exist_ok=True)

# Save encoders
joblib.dump(encoders, "model/encoders.pkl")

# Save processed dataset
df.to_csv("data/ProcessedDataset.csv", index=False)

print("✅ Feature Engineering Completed")
print(df.head())