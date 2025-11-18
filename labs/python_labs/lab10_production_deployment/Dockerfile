FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY solution.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["python3", "solution.py"]
