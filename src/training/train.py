import pandas as pd
import numpy as np
import boto3
import io
import pickle
import torch
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report
)
import tensorflow as tf
from tensorflow import keras

import os

# Always run relative to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# S3 configuration
BUCKET_NAME = "fraud-detection-mlops-rd"
PROCESSED_PREFIX = "data/processed"

# MLflow configuration
EXPERIMENT_NAME = "fraud-detection"
mlflow.set_experiment(EXPERIMENT_NAME)

# Initialize S3 client
s3_client = boto3.client("s3", region_name="us-east-2")

print("Imports done")

def load_processed_data():
    """Load processed train and validation data from S3."""
    print("Loading processed data from S3...")
    
    # Load train data
    train_obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/train.csv"
    )
    train_df = pd.read_csv(io.BytesIO(train_obj["Body"].read()))
    
    # Load validation data
    val_obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/val.csv"
    )
    val_df = pd.read_csv(io.BytesIO(val_obj["Body"].read()))
    
    # Separate features and target
    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]
    
    X_val = val_df.drop(columns=["isFraud"])
    y_val = val_df["isFraud"]
    
    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val


def evaluate_model(model, X_val, y_val, model_type="sklearn"):
    """Evaluate model and return metrics dictionary."""
    
    if model_type == "sklearn":
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
    elif model_type == "tensorflow":
        y_pred_proba = model.predict(X_val).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
    elif model_type == "tabnet":
        y_pred_proba = model.predict_proba(X_val.values)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        "auc_roc": roc_auc_score(y_val, y_pred_proba),
        "f1": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred)
    }
    
    print(f"\nModel Performance:")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    return metrics

def train_xgboost(X_train, X_val, y_train, y_val):
    """Train XGBoost model and log to MLflow."""
    print("\nTraining XGBoost...")
    
    params = {
        "n_estimators": 100, # was 500
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1]),
        "random_state": 42,
        "eval_metric": "auc",
        "early_stopping_rounds": 50
    }
    
    with mlflow.start_run(run_name="xgboost_baseline"):
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, model_type="sklearn")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "xgboost_model")
        
        print(f"XGBoost training complete")
        
    return model, metrics

def train_mlp(X_train, X_val, y_train, y_val):
    """Train MLP neural network and log to MLflow."""
    print("\nTraining MLP Neural Network...")
    
    params = {
        "layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 10, # was 50
        "batch_size": 1024
    }
    
    with mlflow.start_run(run_name="mlp_neural_network"):
        mlflow.log_params(params)
        
        # Build model
        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["auc"]
        )
        
        # Class weights to handle imbalance
        class_weight = {
            0: 1,
            1: len(y_train[y_train==0]) / len(y_train[y_train==1])
        }
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, model_type="tensorflow")
        
        # Log metrics and model
        mlflow.log_metrics(metrics)
        mlflow.tensorflow.log_model(model, "mlp_model")
        
        print("MLP training complete")
        
    return model, metrics


def train_tabnet(X_train, X_val, y_train, y_val):
    """Train TabNet model and log to MLflow."""
    print("\nTraining TabNet...")
    
    from pytorch_tabnet.tab_model import TabNetClassifier
    
    params = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "momentum": 0.02,
        "mask_type": "entmax"
    }
    
    with mlflow.start_run(run_name="tabnet"):
        mlflow.log_params(params)
        
        model = TabNetClassifier(
            **params,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": 2e-3},
            scheduler_params={
                "step_size": 10,
                "gamma": 0.9
            },
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            verbose=10
        )
        
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_metric=["auc"],
            max_epochs=20, # was 100
            patience=5, # was 10
            batch_size=1024,
            virtual_batch_size=128,
            weights=1
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_val, y_val, model_type="tabnet")
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save and log TabNet model
        model.save_model("tabnet_model")
        mlflow.log_artifact("tabnet_model.zip")
        
        print("TabNet training complete")
        
    return model, metrics


def train_autoencoder(X_train, X_val, y_train, y_val):
    """Train autoencoder on normal transactions only for anomaly detection."""
    print("\nTraining Autoencoder...")
    
    params = {
        "encoding_dim": 32,
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "epochs": 10,  # was 50
        "batch_size": 1024,
        "threshold_percentile": 95
    }
    
    with mlflow.start_run(run_name="autoencoder_anomaly_detection"):
        mlflow.log_params(params)
        
        # Train ONLY on normal transactions
        # This is the key insight of autoencoder anomaly detection
        X_train_normal = X_train[y_train == 0]
        
        input_dim = X_train.shape[1]
        
        # Build autoencoder
        inputs = keras.layers.Input(shape=(input_dim,))
        
        # Encoder — compress down
        encoded = keras.layers.Dense(params["hidden_dim"], activation="relu")(inputs)
        encoded = keras.layers.BatchNormalization()(encoded)
        encoded = keras.layers.Dense(params["encoding_dim"], activation="relu")(encoded)
        
        # Decoder — reconstruct back up
        decoded = keras.layers.Dense(params["hidden_dim"], activation="relu")(encoded)
        decoded = keras.layers.BatchNormalization()(decoded)
        decoded = keras.layers.Dense(input_dim, activation="linear")(decoded)
        
        autoencoder = keras.Model(inputs, decoded)
        
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
            loss="mse"
        )
        
        # Train on normal transactions only
        autoencoder.fit(
            X_train_normal, X_train_normal,
            validation_data=(X_val, X_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        # Calculate reconstruction error on validation set
        X_val_reconstructed = autoencoder.predict(X_val)
        reconstruction_errors = np.mean(np.power(X_val.values - X_val_reconstructed, 2), axis=1)
        
        # Set threshold at 95th percentile of reconstruction errors
        threshold = np.percentile(reconstruction_errors, params["threshold_percentile"])
        mlflow.log_metric("reconstruction_threshold", threshold)
        
        # Classify as fraud if reconstruction error exceeds threshold
        y_pred = (reconstruction_errors > threshold).astype(int)
        y_pred_proba = reconstruction_errors / reconstruction_errors.max()
        
        metrics = {
            "auc_roc": roc_auc_score(y_val, y_pred_proba),
            "f1": f1_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred)
        }
        
        print(f"\nAutoencoder Performance:")
        print(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
        print(f"F1:        {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Reconstruction threshold: {threshold:.4f}")
        
        mlflow.log_metrics(metrics)
        mlflow.tensorflow.log_model(autoencoder, "autoencoder_model")
        
        print("Autoencoder training complete")
        
    return autoencoder, metrics, threshold

def main():
    """Train all models and compare results."""
    print("Starting model training pipeline...")
    
    # Load data
    X_train, X_val, y_train, y_val = load_processed_data()
    
    # Train all models
    results = {}
    
    xgb_model, xgb_metrics = train_xgboost(X_train, X_val, y_train, y_val)
    results["xgboost"] = xgb_metrics
    
    mlp_model, mlp_metrics = train_mlp(X_train, X_val, y_train, y_val)
    results["mlp"] = mlp_metrics
    
    tabnet_model, tabnet_metrics = train_tabnet(X_train, X_val, y_train, y_val)
    results["tabnet"] = tabnet_metrics
    
    autoencoder_model, autoencoder_metrics, threshold = train_autoencoder(
        X_train, X_val, y_train, y_val
    )
    results["autoencoder"] = autoencoder_metrics
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'AUC-ROC':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-"*60)
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['auc_roc']:<10.4f} {metrics['f1']:<10.4f} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f}")
    
    # Identify best model by AUC-ROC
    best_model = max(results, key=lambda x: results[x]["auc_roc"])
    print(f"\nBest model: {best_model} with AUC-ROC: {results[best_model]['auc_roc']:.4f}")
    
    print("\nAll models logged to MLflow.")
    print("Open MLflow UI at http://localhost:5000 to compare runs.")

if __name__ == "__main__":
    main()