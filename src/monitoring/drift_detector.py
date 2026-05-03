import boto3
import pandas as pd
import numpy as np
import json
import io
import pickle
import os
from datetime import datetime, timedelta

# Always run relative to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
BUCKET_NAME = "fraud-detection-mlops-rd"
PROCESSED_PREFIX = "data/processed"
DRIFT_THRESHOLD = 0.1  # 10% drift triggers alert

# AWS clients
s3_client = boto3.client("s3", region_name="us-east-2")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")
sns_client = boto3.client("sns", region_name="us-east-2")
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")

# SNS topic ARN
SNS_TOPIC_ARN = "your-sns-topic-arn"

def load_training_distribution():
    """Load reference distribution from training data."""
    print("Loading training data distribution...")
    
    # Load validation set as reference distribution
    val_obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/val.csv"
    )
    val_df = pd.read_csv(io.BytesIO(val_obj["Body"].read()))
    
    # Calculate reference statistics for numeric columns
    numeric_cols = val_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "isFraud"]
    
    reference_stats = {}
    for col in numeric_cols:
        reference_stats[col] = {
            "mean": float(val_df[col].mean()),
            "std": float(val_df[col].std()),
            "min": float(val_df[col].min()),
            "max": float(val_df[col].max()),
            "p25": float(val_df[col].quantile(0.25)),
            "p75": float(val_df[col].quantile(0.75))
        }
    
    # Store fraud rate from validation set
    reference_stats["_fraud_rate"] = float(val_df["isFraud"].mean())
    
    print(f"Reference distribution loaded for {len(numeric_cols)} features")
    print(f"Reference fraud rate: {reference_stats['_fraud_rate']:.2%}")
    
    return reference_stats


def get_recent_predictions(hours=24):
    """Get recent scored transactions from DynamoDB."""
    print(f"Loading predictions from last {hours} hours...")
    
    table = dynamodb.Table("fraud-predictions")
    
    # Scan the table for recent records
    response = table.scan()
    items = response["Items"]
    
    # Handle pagination
    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response["Items"])
    
    if not items:
        print("No predictions found in DynamoDB")
        return None
    
    df = pd.DataFrame(items)
    
    # Convert types
    df["fraud_probability"] = df["fraud_probability"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    
    print(f"Loaded {len(df)} predictions")
    print(f"Current fraud flag rate: {df['is_fraud'].mean():.2%}")
    
    return df


def detect_drift(reference_stats, predictions_df):
    """Compare current prediction distribution against reference."""
    print("\nRunning drift detection...")
    
    drift_results = {}
    drift_detected = False
    
    # Check prediction distribution drift
    if predictions_df is not None and len(predictions_df) > 0:
        current_fraud_rate = predictions_df["is_fraud"].mean()
        reference_fraud_rate = reference_stats["_fraud_rate"]
        
        fraud_rate_drift = abs(current_fraud_rate - reference_fraud_rate)
        drift_results["fraud_rate"] = {
            "reference": reference_fraud_rate,
            "current": current_fraud_rate,
            "drift": fraud_rate_drift,
            "alert": fraud_rate_drift > DRIFT_THRESHOLD
        }
        
        if fraud_rate_drift > DRIFT_THRESHOLD:
            drift_detected = True
            print(f"DRIFT ALERT: Fraud rate drifted by {fraud_rate_drift:.2%}")
            print(f"Reference: {reference_fraud_rate:.2%} Current: {current_fraud_rate:.2%}")
        else:
            print(f"Fraud rate stable: {current_fraud_rate:.2%} (reference: {reference_fraud_rate:.2%})")
    
    return drift_results, drift_detected

def publish_metrics_to_cloudwatch(predictions_df, drift_results):
    """Publish custom metrics to CloudWatch."""
    print("\nPublishing metrics to CloudWatch...")
    
    if predictions_df is None or len(predictions_df) == 0:
        print("No predictions to publish")
        return
    
    current_time = datetime.utcnow()
    
    metrics = [
        {
            "MetricName": "FraudFlagRate",
            "Value": float(predictions_df["is_fraud"].mean()),
            "Unit": "None",
            "Timestamp": current_time
        },
        {
            "MetricName": "TotalPredictions",
            "Value": float(len(predictions_df)),
            "Unit": "Count",
            "Timestamp": current_time
        },
        {
            "MetricName": "AvgFraudProbability",
            "Value": float(predictions_df["fraud_probability"].mean()),
            "Unit": "None",
            "Timestamp": current_time
        }
    ]
    
    # Add drift metric
    if "fraud_rate" in drift_results:
        metrics.append({
            "MetricName": "FraudRateDrift",
            "Value": float(drift_results["fraud_rate"]["drift"]),
            "Unit": "None",
            "Timestamp": current_time
        })
    
    cloudwatch.put_metric_data(
        Namespace="FraudDetection",
        MetricData=metrics
    )
    
    print(f"Published {len(metrics)} metrics to CloudWatch")

def send_drift_alert(drift_results):
    """Send SNS alert when drift is detected."""
    print("\nSending drift alert...")
    
    alert_message = {
        "alert_type": "Model Drift Detected",
        "timestamp": datetime.utcnow().isoformat(),
        "drift_results": drift_results,
        "action_required": "Consider retraining the model"
    }
    
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject="FRAUD MODEL DRIFT ALERT",
        Message=json.dumps(alert_message, indent=2)
    )
    
    print("Drift alert sent via SNS")


def main():
    """Run the full drift detection pipeline."""
    print("Starting drift detection pipeline...")
    print(f"Timestamp: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    # Load reference distribution from training data
    reference_stats = load_training_distribution()
    
    # Get recent predictions from DynamoDB
    predictions_df = get_recent_predictions(hours=24)
    
    # Detect drift
    drift_results, drift_detected = detect_drift(
        reference_stats, 
        predictions_df
    )
    
    # Publish metrics to CloudWatch
    publish_metrics_to_cloudwatch(predictions_df, drift_results)
    
    # Send alert if drift detected
    if drift_detected:
        send_drift_alert(drift_results)
        print("\nDrift detected — alert sent")
    else:
        print("\nNo significant drift detected")
    
    print("=" * 50)
    print("Drift detection complete")

if __name__ == "__main__":
    main()