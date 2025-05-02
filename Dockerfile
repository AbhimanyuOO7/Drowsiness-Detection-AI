# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 --index-url https://pypi.org/simple -r requirements.txt

# Copy the rest of your project files
COPY . .

# Set environment variable to indicate running in Docker
ENV RUNNING_IN_DOCKER=1

# Set environment variables to avoid issues with headless environments
ENV DISPLAY=:99
ENV QT_X11_NO_MITSHM=1

# Run the application
CMD ["python", "drowsinessdetection.py"]