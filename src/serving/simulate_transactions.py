import boto3
import json
import random
import time
import os

# Always run relative to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
STREAM_NAME = "fraud-transactions"
REGION = "us-east-2"

kinesis_client = boto3.client("kinesis", region_name=REGION)

def generate_transaction():
    """Generate a realistic fake transaction."""
    transaction_id = random.randint(1000000, 9999999)
    
    return {
        "TransactionID": transaction_id,
        "TransactionDT": random.randint(86400, 15811131),
        "TransactionAmt": round(random.uniform(1.0, 2000.0), 2),
        "ProductCD": random.choice(["W", "H", "C", "S", "R"]),
        "card1": random.randint(1000, 9999),
        "card2": random.uniform(100, 600),
        "card3": random.uniform(100, 200),
        "card4": random.choice(["visa", "mastercard", "american express", "discover"]),
        "card5": random.uniform(100, 250),
        "card6": random.choice(["debit", "credit"]),
        "addr1": random.randint(100, 500),
        "addr2": random.randint(10, 100),
        "dist1": random.uniform(0, 10000),
        "P_emaildomain": random.choice([
            "gmail.com", "yahoo.com", "hotmail.com", 
            "anonymous.com", "unknown.com"
        ]),
        "R_emaildomain": random.choice([
            "gmail.com", "yahoo.com", "hotmail.com",
            "anonymous.com", "unknown.com"
        ]),
    }

def generate_suspicious_transaction():
    """Generate a transaction with fraud-like characteristics."""
    transaction = generate_transaction()
    
    # Fraud signals
    transaction["TransactionAmt"] = round(random.uniform(1000, 5000), 2)
    transaction["card4"] = "unknown"
    transaction["P_emaildomain"] = "anonymous.com"
    transaction["R_emaildomain"] = "unknown.com"
    transaction["TransactionDT"] = random.randint(0, 3600)  # Early morning
    
    return transaction

def send_transaction(transaction):
    """Send a single transaction to Kinesis."""
    response = kinesis_client.put_record(
        StreamName=STREAM_NAME,
        Data=json.dumps(transaction),
        PartitionKey=str(transaction["TransactionID"])
    )
    return response

def main():
    print("Starting transaction simulation...")
    print(f"Sending to Kinesis stream: {STREAM_NAME}")
    print("-" * 50)
    
    for i in range(20):
        # Send mostly normal, some suspicious
        if random.random() < 0.2:
            transaction = generate_suspicious_transaction()
            label = "SUSPICIOUS"
        else:
            transaction = generate_transaction()
            label = "normal"
        
        response = send_transaction(transaction)
        print(f"Sent {label} transaction {transaction['TransactionID']} "
              f"— ${transaction['TransactionAmt']} "
              f"— Shard: {response['ShardId']}")
        
        time.sleep(1)
    
    print("-" * 50)
    print("Simulation complete. Check DynamoDB for results.")
    print("Check your email for any fraud alerts from SNS.")

if __name__ == "__main__":
    main()