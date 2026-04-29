import boto3
import mlflow
import mlflow.tensorflow
import os
import tempfile

# Always run relative to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
BUCKET_NAME = "fraud-detection-mlops-rd"
MODEL_PREFIX = "models"

# Point MLflow at our local database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

s3_client = boto3.client("s3", region_name="us-east-2")

def get_best_model():
    """Get the best MLP model run from MLflow."""
    print("Finding best MLP model in MLflow...")
    
    client = mlflow.tracking.MlflowClient()
    
    # Get the fraud-detection experiment
    experiment = client.get_experiment_by_name("fraud-detection")
    
    # Search for the best MLP run by AUC-ROC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.mlflow.runName = 'mlp_neural_network'",
        order_by=["metrics.auc_roc DESC"],
        max_results=1
    )
    
    best_run = runs[0]
    print(f"Best MLP run ID: {best_run.info.run_id}")
    print(f"AUC-ROC: {best_run.data.metrics['auc_roc']:.4f}")
    
    return best_run

def export_model_to_s3(run):
    """Download model from MLflow and upload to S3."""
    print("Exporting model to S3...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download model artifacts from MLflow
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(
            run.info.run_id,
            "mlp_model",
            tmp_dir
        )
        
        # Walk through downloaded files and upload each to S3
        model_dir = os.path.join(tmp_dir, "mlp_model")
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                # Create S3 key preserving folder structure
                relative_path = os.path.relpath(local_path, tmp_dir)
                s3_key = f"{MODEL_PREFIX}/{relative_path}".replace("\\", "/")
                
                s3_client.upload_file(local_path, BUCKET_NAME, s3_key)
                print(f"Uploaded: {s3_key}")
    
    print("Model exported to S3 successfully")

def main():
    run = get_best_model()
    export_model_to_s3(run)
    print(f"\nModel available at s3://{BUCKET_NAME}/{MODEL_PREFIX}/mlp_model/")

if __name__ == "__main__":
    main()