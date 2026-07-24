import pandas as pd
df=pd.read_csv('data/Carbon Emission.csv')
print(df.head())
print(df.columns)
print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)
# Replace missing vehicle types
df["Vehicle Type"] = df["Vehicle Type"].fillna("No Vehicle")
print(df.isnull().sum())
columns = [
    "Transport",
    "Vehicle Type",
    "Vehicle Monthly Distance Km",
    "Energy efficiency",
    "Waste Bag Weekly Count",
    "Recycling",
    "Monthly Grocery Bill",
    "Frequency of Traveling by Air",
    "CarbonEmission"
]

df = df[columns]
df.drop_duplicates(inplace=True)
df.to_csv("data/CleanedDataset.csv", index=False)

print("Dataset Saved Successfully")