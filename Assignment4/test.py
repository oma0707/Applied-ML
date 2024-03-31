# test.py

import pytest
import joblib
import os
import requests
import threading
import subprocess
import time
from score import score

@pytest.fixture
def example_text():
    return "URGENT: Exclusive Insurance Offer! Act Now to Secure Your Future! Limited Time Only: Claim Your Policy Before It's Too Late! Don't Miss Out on This Life-Saving Opportunity!"


@pytest.fixture
def example_threshold():
    return 0.5


# Function to send sample requests to container and verify that it has been set up correctly 
def wait_for_container_ready():
    max_retries = 10
    retry_delay = 30  # seconds

    for _ in range(max_retries):
        try:
            # Send a sample request to the container to check if it's ready
            payload = {
                "text": "sample_text", 
                "threshold":0.5
                }
            response = requests.post("http://127.0.0.1:5000/score", json=payload, timeout=15)
            if response.status_code == 200:
                print("Container is ready")
                return True
        except Exception as e:
            print(f"Error checking container status: {e}")

        print("Container is not ready yet, retrying...")
        time.sleep(retry_delay)

    print("Max retries exceeded, container is not ready")
    return False


def run_image():
    subprocess.run(["docker", "build", "-t", "spam-classifier", "."])

def test_docker(example_text, example_threshold):
    # Run Docker container in a separate thread
    image_thread = threading.Thread(target=run_image)
    image_thread.start()
    time.sleep(75)
    print('Building Container')
    subprocess.run(["docker", "run", "-d", "-p", "5000:5000", "--name", "spam-container", "spam-classifier"])
    # Wait for the container to be ready
    
    if wait_for_container_ready():
        print("Test passed!")
        with open("test_results.txt", "a") as f:
            f.write("Test passed!\n")
    else:
        print("Test failed!")
        with open("test_results.txt", "a") as f:
            f.write("Test failed!\n")

    
    # Test the response from the localhost endpoint
    payload = {
        'text': example_text,
        'threshold': example_threshold
    }
    print('Sending Request')
    response = requests.post('http://127.0.0.1:5000/score', json=payload)
    data = response.json()
    prediction = data['prediction']
    propensity = data['propensity']

    # Check if the response contains prediction and propensity
    assert 'prediction' in data
    assert 'propensity' in data

    prediction_str = "spam" if prediction else "non-spam"

    # Print the result
    print(f'The text to be tested was "{example_text}"')
    print(f'It was classified as {prediction_str} with score {propensity}')
    
    
    # Close the Docker container
    subprocess.run(["docker", "stop", "spam-container"])
    subprocess.run(["docker", "rm", "spam-container"])
    subprocess.run(["docker", "rmi", "spam-classifier"])