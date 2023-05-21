# Use a base image with Python installed
FROM python:3.9

# Set the working directory
WORKDIR /app

# Create a virtual environment and activate it
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the requirements.txt file to the working directory
COPY requirements.txt .
COPY similarity.py ./utils/
COPY model .
COPY data/clothing_updated.csv ./data/
COPY data/embeddings.npy ./data/

# Create a virtual environment and activate it
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"

# Set the command to run when the container starts
CMD ["python", "main.py"]
