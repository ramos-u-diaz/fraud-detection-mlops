# Use a newer Python base image with modern GCC
FROM public.ecr.aws/lambda/python:3.12

# Install build tools
RUN dnf install -y gcc gcc-c++ make cmake

# Copy requirements first for Docker layer caching
COPY requirements_lambda.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Copy Lambda function code
COPY src/serving/lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the handler
CMD ["lambda_function.lambda_handler"]