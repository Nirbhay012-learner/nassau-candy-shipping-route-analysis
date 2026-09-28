"""Data cleaning for the Nassau Candy Distributor project."""
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
    df["Lead Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    # Validation only; source dates are intentionally not altered.
    if df.isnull().sum().sum() > 0:
        print("Warning: missing values are present.")
    if (df["Lead Time"] < 0).any():
        print("Warning: negative lead times are present.")
    return df

if __name__ == "__main__":
    df = clean_data(load_data("../data/shipping_data.csv"))
    print("Data cleaning completed.")
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
