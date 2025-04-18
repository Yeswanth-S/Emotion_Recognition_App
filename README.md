
# Emotion Detection App

## 🚀 Overview
The Emotion Detection App is a web-based application that uses deep learning to recognize human emotions from facial expressions. Built with **Flask**, **OpenCV**, and **TensorFlow**, the app supports multiple input methods including image upload, video upload, and real-time webcam feed.

It leverages a pre-trained convolutional neural network model to classify facial emotions in various input formats.

## 🎯 Features
- 📷 **Image Upload**: Detect emotions in static images.
- 🎞️ **Video Upload**: Analyze emotions frame by frame in a video.
- 🎥 **Webcam Mode**: Real-time emotion detection using your webcam.
- 🔴 **Live Feed**: Continuous facial emotion recognition through a live video stream.

<details>
<summary>📁 Project Structure</summary>

```
Emotion-Detection-App/
│
├── app.py                     # Main Flask application
├── Dockerfile                 # Docker configuration file
├── Main.py                    # Model training script
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── app/                       # Flask app directory
│   ├── static/                # Static files (CSS, JS)
│   │   └── css/
│   │       ├── index.css      # Styles for the homepage
│   │       └── styles.css     # General styles
│   │
│   └── templates/             # HTML templates for the app
│       ├── 404.html           # Custom 404 page
│       ├── index.html         # Homepage template
│       ├── livefeed.html      # Live feed page
│       ├── sidebar.html       # Sidebar template
│       ├── upload_image.html  # Image upload page
│       ├── upload_video.html  # Video upload page
│       └── use_webcam.html    # Webcam usage page
│
├── data/                      # Data directory (not included)
│   └── fer2013.csv            # FER-2013 emotion dataset
│
├── models/                    # Machine learning models and configuration (not included)
│   ├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
│   ├── model.h5               # Pre-trained model weights
│   └── model.json             # Model architecture
│
└── uploads/                   # Temporary file storage for uploaded files
```

</details>

<details>
<summary>🛠️ Getting Started</summary>

### 1. Clone the Repository
```bash
git clone https://github.com/YESWANTH-S/Emotion_Recognition_App.git 
cd Emotion_Recognition_App
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the Model Files
Place the following files in the root project directory:
- `model.json`
- `model.h5`

> ⚠️ These files are **not included** in the repository. You must add them manually via running the `Main.py`.

### 4. Run the Application
```bash
python app.py
```

### 5. Open in Browser
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

</details>

<details>
<summary>🧪 How to Use</summary>

| Feature       | URL Path           | Description                              |
|---------------|--------------------|------------------------------------------|
| Upload Image  | `/upload_image`    | Upload an image to detect emotions       |
| Upload Video  | `/upload_video`    | Analyze a video frame-by-frame           |
| Use Webcam    | `/use_webcam`      | Real-time emotion detection via webcam   |
| Live Feed     | `/livefeed`        | Continuous emotion recognition stream    |

</details>

<details>
<summary>📦 Dependencies</summary>

- Flask
- OpenCV
- TensorFlow
- NumPy

</details>

<details>
<summary>🐳 Docker (Optional)</summary>

### Build and Run the App using Docker

```bash
docker build -t emotion-detector .
docker run -p 5000:5000 emotion-detector
```

> Make sure `model.h5` and `model.json` are in the root directory before building the image.

</details>

<details>
<summary>🤖 Model</summary>

The model is trained to detect the following emotions:
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

Note: Model training can be initiated by running the Main.py file.
</details>

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).  
Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or a pull request.
