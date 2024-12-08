# Use the official Python image from Docker Hub as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /appsdevelopmentprofile/SPE_Demo/streamlit_app

# Copy requirements.txt into the container at /app
COPY requirements.txt .

# Install any necessary dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app will run on (Streamlit default is 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]