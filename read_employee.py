import pandas as pd

csv_path = "data/Employee.csv"
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("Column counts (non-null values):")
print(df.count())

print("\nUnique value counts per column:")
print(df.nunique())
