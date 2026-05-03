import json
import boto3
import numpy as np
import os
import pickle
import io
import logging

# Set up logging — this goes to CloudWatch automatically
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration — loaded from environment variables
# We never hardcode these in Lambda functions
BUCKET_NAME = os.environ.get("BUCKET_NAME", "fraud-detection-mlops-rd")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/mlp_model")
PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "data/processed")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "fraud-predictions")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.5"))

# Initialize AWS clients
s3_client = boto3.client("s3", region_name="us-east-2")
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
sns_client = boto3.client("sns", region_name="us-east-2")

# Global model variable — loaded once per container, reused across invocations
model = None
scaler = None
encoders = None


def load_model():
    """Load model, scaler and encoders from S3 into memory."""
    global model, scaler, encoders
    
    if model is not None:
        logger.info("Model already loaded, reusing...")
        return
    
    logger.info("Loading model from S3...")
    
    import tensorflow as tf
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_dir = os.path.join(tmp_dir, "mlp_saved_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Download all SavedModel files
        s3_prefix = "models/mlp_saved_model"
        paginator = s3_client.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                relative_path = key[len(s3_prefix)+1:]
                local_path = os.path.join(model_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_client.download_file(BUCKET_NAME, key, local_path)
        
        model = tf.saved_model.load(model_dir)
        logger.info("Model loaded successfully")
    
    # Load scaler
    scaler_obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/scaler.pkl"
    )
    scaler = pickle.loads(scaler_obj["Body"].read())
    logger.info("Scaler loaded successfully")
    
    # Load encoders
    encoders_obj = s3_client.get_object(
        Bucket=BUCKET_NAME,
        Key=f"{PROCESSED_PREFIX}/encoders.pkl"
    )
    encoders = pickle.loads(encoders_obj["Body"].read())
    logger.info("Encoders loaded successfully")


def engineer_features(transaction):
    """Apply same feature engineering as training pipeline."""
    
    # Transaction amount features
    transaction["TransactionAmt_log"] = np.log1p(
        transaction.get("TransactionAmt", 0)
    )
    transaction["TransactionAmt_decimal"] = (
        transaction.get("TransactionAmt", 0) % 1
    )
    
    # Time-based features
    dt = transaction.get("TransactionDT", 0)
    transaction["hour"] = (dt / 3600) % 24
    transaction["day_of_week"] = (dt / (3600 * 24)) % 7
    transaction["is_night"] = int(
        transaction["hour"] >= 22 or transaction["hour"] <= 6
    )
    transaction["is_weekend"] = int(transaction["day_of_week"] >= 5)
    
    # Amount diff from card average — use 0 as default at inference time
    # In production this would query a feature store for the card's history
    transaction["amt_diff_from_card_avg"] = 0
    
    return transaction


def encode_and_scale(transaction_df):
    """Apply saved encoders and scaler to incoming transaction."""
    
    # Apply label encoders to categorical columns
    for col, encoder in encoders.items():
        if col in transaction_df.columns:
            try:
                transaction_df[col] = encoder.transform(
                    transaction_df[col].astype(str)
                )
            except ValueError:
                transaction_df[col] = 0
    
    # Add missing columns with 0 — transaction may not have all features
    expected_cols = scaler.feature_names_in_
    for col in expected_cols:
        if col not in transaction_df.columns:
            transaction_df[col] = 0
    
    # Keep only expected columns in correct order
    transaction_df = transaction_df[expected_cols]
    
    # Apply scaler
    transaction_df[expected_cols] = scaler.transform(transaction_df[expected_cols])
    
    return transaction_df


def lambda_handler(event, context):
    """Main Lambda handler — triggered by Kinesis stream."""
    
    # Load model on first invocation
    load_model()
    
    results = []
    
    # Kinesis sends records in batches
    for record in event["Records"]:
        try:
            # Decode Kinesis record
            import base64
            payload = json.loads(
                base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
            )
            
            transaction_id = payload.get("TransactionID", "unknown")
            logger.info(f"Scoring transaction: {transaction_id}")
            
            # Engineer features
            payload = engineer_features(payload)
            
            # Convert to DataFrame for sklearn compatibility
            import pandas as pd
            transaction_df = pd.DataFrame([payload])
            
            # Drop TransactionID before scoring
            if "TransactionID" in transaction_df.columns:
                transaction_df = transaction_df.drop(columns=["TransactionID"])
            
            # Remove target column if present
            if "isFraud" in transaction_df.columns:
                transaction_df = transaction_df.drop(columns=["isFraud"])
            
            # Encode and scale
            transaction_df = encode_and_scale(transaction_df)
            
            # Score transaction
            import tensorflow as tf
            input_tensor = tf.constant(transaction_df.values, dtype=tf.float32)
            infer = model.signatures["serving_default"]
            output = infer(input_tensor)
            fraud_probability = float(list(output.values())[0].numpy()[0][0])
            is_fraud = fraud_probability >= FRAUD_THRESHOLD
            
            logger.info(f"Transaction {transaction_id}: {fraud_probability:.4f}")
            
            # Store result in DynamoDB
            table = dynamodb.Table(DYNAMODB_TABLE)
            from decimal import Decimal

            table.put_item(Item={
                "TransactionID": str(transaction_id),
                "fraud_probability": Decimal(str(round(fraud_probability, 4))),
                "is_fraud": is_fraud,
                "threshold_used": Decimal(str(FRAUD_THRESHOLD)),
                "timestamp": str(record["kinesis"]["approximateArrivalTimestamp"])
            })
            
            # Send SNS alert for high risk transactions
            if is_fraud and SNS_TOPIC_ARN:
                sns_client.publish(
                    TopicArn=SNS_TOPIC_ARN,
                    Subject=f"FRAUD ALERT — Transaction {transaction_id}",
                    Message=json.dumps({
                        "TransactionID": transaction_id,
                        "fraud_probability": round(fraud_probability, 4),
                        "message": "High risk transaction detected"
                    }, indent=2)
                )
                logger.info(f"SNS alert sent for transaction {transaction_id}")
            
            results.append({
                "TransactionID": transaction_id,
                "fraud_probability": round(fraud_probability, 4),
                "is_fraud": is_fraud
            })
            
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}")
            continue
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": len(results),
            "results": results
        })
    }