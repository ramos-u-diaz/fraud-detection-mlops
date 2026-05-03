import streamlit as st
import boto3
import pandas as pd
import numpy as np
from decimal import Decimal
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide"
)

# AWS configuration
BUCKET_NAME = "fraud-detection-mlops-rd"
REGION = "us-east-2"
DYNAMODB_TABLE = "fraud-predictions"

# Initialize AWS clients
@st.cache_resource
def get_aws_clients():
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)
    return dynamodb, cloudwatch

dynamodb, cloudwatch = get_aws_clients()


@st.cache_data(ttl=30)
def load_predictions():
    """Load predictions from DynamoDB with 30 second cache."""
    table = dynamodb.Table(DYNAMODB_TABLE)
    
    response = table.scan()
    items = response["Items"]
    
    while "LastEvaluatedKey" in response:
        response = table.scan(
            ExclusiveStartKey=response["LastEvaluatedKey"]
        )
        items.extend(response["Items"])
    
    if not items:
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    df["fraud_probability"] = df["fraud_probability"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    
    return df.sort_values("timestamp", ascending=False)

@st.cache_data(ttl=30)
def load_cloudwatch_metrics():
    """Load custom metrics from CloudWatch."""
    try:
        response = cloudwatch.get_metric_statistics(
            Namespace="FraudDetection",
            MetricName="FraudFlagRate",
            StartTime=datetime.utcnow() - pd.Timedelta(hours=24),
            EndTime=datetime.utcnow(),
            Period=3600,
            Statistics=["Average"]
        )
        return response.get("Datapoints", [])
    except Exception:
        return []
    
def main():
    """Main dashboard layout."""
    
    # Header
    st.title("🔍 Real-Time Fraud Detection Dashboard")
    st.markdown("Live monitoring of payment transaction scoring pipeline")
    st.divider()
    
    # Load data
    with st.spinner("Loading data..."):
        predictions_df = load_predictions()
    
    if predictions_df.empty:
        st.warning("No predictions found in DynamoDB. Run the transaction simulator first.")
        return
    
    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions Scored",
            value=len(predictions_df)
        )
    
    with col2:
        fraud_count = predictions_df["is_fraud"].sum()
        st.metric(
            label="Flagged as Fraud",
            value=int(fraud_count)
        )
    
    with col3:
        fraud_rate = predictions_df["is_fraud"].mean()
        st.metric(
            label="Fraud Flag Rate",
            value=f"{fraud_rate:.2%}"
        )
    
    with col4:
        avg_prob = predictions_df["fraud_probability"].mean()
        st.metric(
            label="Avg Fraud Probability",
            value=f"{avg_prob:.4f}"
        )
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.subheader("Fraud Probability Distribution")
        fig = px.histogram(
            predictions_df,
            x="fraud_probability",
            nbins=20,
            color_discrete_sequence=["#e74c3c"],
            labels={"fraud_probability": "Fraud Probability"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.subheader("Fraud vs Legitimate Transactions")
        fraud_counts = predictions_df["is_fraud"].value_counts().reset_index()
        fraud_counts.columns = ["is_fraud", "count"]
        fraud_counts["label"] = fraud_counts["is_fraud"].map({
            True: "Fraud", 
            False: "Legitimate"
        })
        fig = px.pie(
            fraud_counts,
            values="count",
            names="label",
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Recent transactions table
    st.subheader("Recent Transactions")
    display_df = predictions_df[[
        "TransactionID", 
        "fraud_probability", 
        "is_fraud", 
        "datetime"
    ]].head(20).copy()
    
    display_df.columns = [
        "Transaction ID", 
        "Fraud Probability", 
        "Flagged as Fraud", 
        "Timestamp"
    ]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Auto refresh
    st.caption("Dashboard refreshes every 30 seconds")
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()