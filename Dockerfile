# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on (default is 8000)
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]