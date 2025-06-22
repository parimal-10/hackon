# 1. Base Python image
FROM python:3.9-slim

# 2. Create & switch to working dir
WORKDIR /app

# 3. Copy & install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app code & model files
COPY . .

# 5. Expose the port your Flask app listens on
EXPOSE 5000

# 6. Default command
CMD ["python", "app.py"]
