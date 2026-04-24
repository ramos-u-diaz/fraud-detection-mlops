import pandas as pd
import numpy as np
import boto3
import io
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# S3 configuration
BUCKET_NAME = "fraud-detection-mlops-rd"
RAW_PREFIX = "data/raw"
PROCESSED_PREFIX = "data/processed"

# Initialize S3 client
s3_client = boto3.client("s3", region_name="us-east-2")

def load_data_from_s3():
    """Load transaction and identity data from S3 and merge them."""
    print("Loading training data from S3...")

    # Load transaction data
    trans_obj = s3_client.get_object(
        Bucket=BUCKET_NAME, 
        Key=f"{RAW_PREFIX}/train_transaction.csv"
    )
    df_trans = pd.read_csv(io.BytesIO(trans_obj["Body"].read()))

    # Load identity data
    identity_obj = s3_client.get_object(
        Bucket=BUCKET_NAME, 
        Key=f"{RAW_PREFIX}/train_identity.csv"
    )
    df_identity = pd.read_csv(io.BytesIO(identity_obj["Body"].read()))

    # Merge on TransactionID — left join keeps all transactions
    # even those without identity data
    df = df_trans.merge(df_identity, on="TransactionID", how="left")
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Fraud rate: {df['isFraud'].mean():.2%}")
    
    return df

def handle_missing_values(df):
    """Handle missing values for numeric and categorical columns."""
    print("Handling missing values...")
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Fill numeric missing values with median
    # Median is better than mean for fraud data — outliers skew the mean
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical missing values with the string "Unknown"
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")
    
    print(f"Missing values remaining: {df.isnull().sum().sum()}")
    
    return df

def encode_categoricals(df):
    """Encode categorical columns to numeric using LabelEncoder."""
    print("Encoding categorical variables...")
    
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"Encoded {len(categorical_cols)} categorical columns")
    
    return df, encoders


def engineer_features(df):
    """Create new features from existing ones."""
    print("Engineering features...")
    
    # Transaction amount features
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = df["TransactionAmt"] % 1
    
    # Time-based features
    # TransactionDT is seconds elapsed — not a real timestamp
    df["hour"] = (df["TransactionDT"] / 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] / (3600 * 24)) % 7
    
    # Fraud tends to happen at odd hours and on weekends
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Transaction amount differs from card's typical amount
    card_avg = df.groupby("card1")["TransactionAmt"].transform("mean")
    df["amt_diff_from_card_avg"] = df["TransactionAmt"] - card_avg
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    
    return df

def scale_and_split(df):
    """Scale numeric features and split into train/validation sets."""
    print("Scaling and splitting data...")
    
    # Separate features and target
    X = df.drop(columns=["isFraud", "TransactionID"])
    y = df["isFraud"]
    
    # Scale numeric columns
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # Important — maintains fraud ratio in both splits
    )
    
    print(f"Train set: {X_train.shape}, Fraud rate: {y_train.mean():.2%}")
    print(f"Val set: {X_val.shape}, Fraud rate: {y_val.mean():.2%}")
    
    return X_train, X_val, y_train, y_val, scaler

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance in training data only."""
    print("Applying SMOTE to training data...")
    print(f"Before SMOTE - Fraud: {y_train.sum()}, Non-fraud: {(y_train==0).sum()}")
    
    smote = SMOTE(random_state=42, sampling_strategy=0.3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE - Fraud: {y_train_resampled.sum()}, Non-fraud: {(y_train_resampled==0).sum()}")
    
    return X_train_resampled, y_train_resampled

def save_to_s3(X_train, X_val, y_train, y_val, scaler, encoders):
    """Save processed data and artifacts to S3."""
    print("Saving processed data to S3...")
    
    import pickle
    
    # Combine features and target for saving
    train_df = X_train.copy()
    train_df["isFraud"] = y_train.values
    
    val_df = X_val.copy()
    val_df["isFraud"] = y_val.values
    
    # Save train and validation sets to S3
    for df, name in [(train_df, "train"), (val_df, "val")]:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=f"{PROCESSED_PREFIX}/{name}.csv",
            Body=buffer.getvalue()
        )
        print(f"Saved {name}.csv to S3")
    
    # Save scaler
    scaler_buffer = io.BytesIO()
    pickle.dump(scaler, scaler_buffer)
    scaler_buffer.seek(0)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/scaler.pkl",
        Body=scaler_buffer.getvalue()
    )
    print("Saved scaler to S3")
    
    # Save encoders
    encoders_buffer = io.BytesIO()
    pickle.dump(encoders, encoders_buffer)
    encoders_buffer.seek(0)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/encoders.pkl",
        Body=encoders_buffer.getvalue()
    )
    print("Saved encoders to S3")
    
    print("All artifacts saved to S3 successfully")


def main():
    """Run the full preprocessing pipeline."""
    print("Starting preprocessing pipeline...")
    
    # Load data
    df = load_data_from_s3()
    
    # Preprocess
    df = handle_missing_values(df)
    df, encoders = encode_categoricals(df)
    df = engineer_features(df)
    
    # Split and scale
    X_train, X_val, y_train, y_val, scaler = scale_and_split(df)
    
    # Handle class imbalance
    X_train, y_train = apply_smote(X_train, y_train)
    
    # Save everything to S3
    save_to_s3(X_train, X_val, y_train, y_val, scaler, encoders)
    
    print("Preprocessing pipeline complete.")

if __name__ == "__main__":
    main()