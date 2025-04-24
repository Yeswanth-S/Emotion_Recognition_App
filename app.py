import os
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import base64

# Get the directory where the script is located.
script_dir = os.path.dirname(os.path.realpath(__file__))

# Set the path to the templates folder
templates_folder = os.path.join(script_dir, 'app', 'templates')

# Set the path to the static folder 
static_folder = os.path.join(script_dir, 'app', 'static')

# Set the path to the models folder
models_folder = os.path.join(script_dir, 'models')

# Initialize Flask app with the custom template and static folder paths
app = Flask(__name__, template_folder=templates_folder, static_folder=static_folder)

# Build the relative path to model.json, model.h5, and haarcascade_frontalface_default.xml
model_json_path = os.path.join(models_folder, 'model.json')
model_weights_path = os.path.join(models_folder, 'model.h5')
face_cascade_path = os.path.join(models_folder, 'haarcascade_frontalface_default.xml')

# Load the model and weights
with open(model_json_path, 'r') as json_file:
    model = model_from_json(json_file.read())
model.load_weights(model_weights_path)

# Load face detection model
face_cascade = cv2.CascadeClassifier(face_cascade_path)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect and classify emotion from image
def detect_emotion_from_image(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    emotions = []
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (48, 48))

        img_array = img_to_array(face_roi_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]
        confidence = np.max(predictions[0]) * 100
        emotions.append({'x': x, 'emotion': predicted_emotion, 'confidence': confidence})

    # Sort detected faces by x-coordinate for left-to-right detection
    emotions_sorted = sorted(emotions, key=lambda item: item['x'])

    # Remove 'x' key from sorted emotions
    for emotion in emotions_sorted:
        del emotion['x']

    return emotions_sorted

# Function to process video and detect emotions
def detect_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        frame_result = []  # To hold emotions for this frame

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (48, 48))

            img_array = img_to_array(face_roi_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[max_index]
            confidence = np.max(predictions[0]) * 100
            
            frame_result.append({'emotion': predicted_emotion, 'confidence': confidence})

        frame_emotions.append(frame_result)

    cap.release()
    return frame_emotions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image')
def upload_image_page():
    return render_template('upload_image.html')

@app.route('/upload_video')
def upload_video_page():
    return render_template('upload_video.html')

@app.route('/use_webcam')
def use_webcam_page():
    return render_template('use_webcam.html')

@app.route('/livefeed')
def livefeed_page():
    return render_template('livefeed.html')

# Add custom 404 handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    temp_file_path = os.path.join('uploads', filename)
    file.save(temp_file_path)

    # Detect emotions
    image = cv2.imread(temp_file_path)
    if image is None:
        os.remove(temp_file_path)
        return jsonify({'error': 'Invalid image file'}), 400

    emotions = detect_emotion_from_image(image)
    os.remove(temp_file_path)

    return jsonify(emotions)  # will return [] if no face detected (frontend handles it)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    temp_file_path = os.path.join('uploads', filename)
    file.save(temp_file_path)

    frame_emotions = detect_emotion_from_video(temp_file_path)
    os.remove(temp_file_path)

    summary = {}
    for frame in frame_emotions:
        for emotion in frame:
            summary[emotion['emotion']] = summary.get(emotion['emotion'], 0) + 1

    summary_output = [{'emotion': key, 'count': value} for key, value in summary.items()]

    return jsonify({
        'summary': summary_output,
        'details': frame_emotions
    })

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data received'}), 400

    try:
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        np_img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        emotions = detect_emotion_from_image(img)
        return jsonify(emotions)  # returns [] if no faces detected

    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)