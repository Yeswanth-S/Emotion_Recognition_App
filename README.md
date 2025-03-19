# Emotion Detection App

## Overview
This is an emotion detection app built using Flask, OpenCV, and TensorFlow. It allows users to upload images, videos, or use a live feed to detect emotions. The app uses a pre-trained deep learning model to classify emotions in faces.

## Features
- Upload an image to detect emotions.
- Upload a video to analyze emotions frame by frame.
- Use your webcam to detect emotions in real-time.
- View live feed for continuous emotion detection.

## Project Structure
- `app.py`: Main Flask application file.
- `uploads/`: Temporary storage for uploaded files.
- `static/`: Folder containing static files (CSS, JS).
- `templates/`: HTML templates for rendering pages.
- `model.json`: Pre-trained model architecture (not included).
- `model.h5`: Pre-trained model weights (not included).

## How to Run
1. Clone this repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Place the `model.json` and `model.h5` files in the project directory.
4. Run the Flask app: `python app.py`.
5. Open your browser and go to `http://127.0.0.1:5000`.

## Dependencies
- Flask
- OpenCV
- TensorFlow
- NumPy

## Usage
- **Upload Image**: Navigate to `/upload_image` and upload an image for emotion detection.
- **Upload Video**: Navigate to `/upload_video` and upload a video for frame-by-frame emotion analysis.
- **Use Webcam**: Navigate to `/use_webcam` to use your webcam for real-time emotion detection.
- **Live Feed**: Navigate to `/livefeed` for live emotion detection on a continuous video stream.
