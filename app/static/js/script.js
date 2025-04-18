// Live Feed Emotion Detection

let videoStream;

// Start webcam and live processing on load (for livefeed page)
window.addEventListener('load', () => {
    if (document.getElementById('video') && document.getElementById('canvas')) {
        startLiveFeed();
    } else if (document.getElementById('video') && document.getElementById('videoContainer')) {
        startWebcam();
    }
});

function startLiveFeed() {
    const video = document.getElementById('video');
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoStream = stream;
            video.srcObject = stream;
            video.play();
            processLiveVideo();
        })
        .catch(err => {
            console.error("Error accessing the webcam:", err);
        });
}

function processLiveVideo() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    setInterval(() => {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg');

        fetch('/detect_frame', {
            method: 'POST',
            body: JSON.stringify({ image: frameData }),
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(res => res.json())
            .then(displayLiveResults)
            .catch(err => console.error('Error:', err));
    }, 1000);
}

function displayLiveResults(data) {
    const result = document.getElementById('result');
    result.innerHTML = '';

    if (data.length === 0) {
        result.innerHTML = '<p>No faces detected.</p>';
        return;
    }

    let resultText = "<h3>Detected Emotions (Left to Right):</h3>";
    data.forEach((emotion, i) => {
        resultText += `<p>Person ${i + 1}: Emotion: <strong>${emotion.emotion}</strong>, Confidence: ${emotion.confidence.toFixed(1)}%</p>`;
    });
    resultText += "<p>Note: Persons are ordered from left to right.</p>";
    result.innerHTML = resultText;
}


// Image Upload Emotion Detection

document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('imageUpload');
    const videoInput = document.getElementById('videoUpload');

    if (imageInput) {
        imageInput.addEventListener('change', handleImageUpload);
    }

    if (videoInput) {
        videoInput.addEventListener('change', handleVideoUpload);
    }
});

function handleImageUpload() {
    const input = document.getElementById('imageUpload');
    const file = input.files[0];
    const fileNameDisplay = document.getElementById('fileName');

    if (file) {
        const fileName = file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name;
        fileNameDisplay.innerHTML = `Selected file: ${fileName}`;
        uploadImage(file);
    } else {
        fileNameDisplay.innerHTML = 'No file selected';
    }
}

function uploadImage(file) {
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const formData = new FormData();
    formData.append('file', file);

    loading.style.display = 'block';
    result.innerHTML = '';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            loading.style.display = 'none';
            if (data.length === 0) {
                result.innerHTML = '<p>No faces detected in the image.</p>';
            } else {
                let resultText = "<h3>Detected Emotion:</h3>";
                data.forEach(emotion => {
                    resultText += `<p>Emotion: <strong>${emotion.emotion}</strong>, Confidence: ${emotion.confidence.toFixed(1)}%</p>`;
                });
                result.innerHTML = resultText;
            }
        })
        .catch(err => {
            loading.style.display = 'none';
            result.innerHTML = '<p>An error occurred while processing the image.</p>';
            console.error('Error:', err);
        });
}

// Video Upload Emotion Detection

function handleVideoUpload() {
    const input = document.getElementById('videoUpload');
    const file = input.files[0];
    const fileNameDisplay = document.getElementById('fileName');

    if (file) {
        const fileName = file.name.length > 20 ? file.name.substring(0, 17) + '...' : file.name;
        fileNameDisplay.innerHTML = `Selected file: ${fileName}`;
        uploadVideo(file);
    } else {
        fileNameDisplay.innerHTML = 'No file selected';
    }
}

function uploadVideo(file) {
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const formData = new FormData();
    formData.append('file', file);

    loading.style.display = 'block';
    result.innerHTML = '';

    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            loading.style.display = 'none';
            if (data.details.length === 0) {
                result.innerHTML = '<p>No faces detected in the video.</p>';
                return;
            }

            let summary = "<h3>Summary of Detected Emotions:</h3>";
            data.summary.forEach(emotion => {
                summary += `<p>${emotion.emotion}: <strong>${emotion.count}</strong></p>`;
            });

            let details = "<h3>Detailed Frame Results:</h3>";
            data.details.forEach((frame, i) => {
                details += `<p><strong>Frame ${i + 1}:</strong></p>`;
                frame.forEach(emotion => {
                    details += `<p>Emotion: <strong>${emotion.emotion}</strong>, Confidence: ${emotion.confidence.toFixed(1)}%</p>`;
                });
            });

            result.innerHTML = summary + details;
        })
        .catch(err => {
            loading.style.display = 'none';
            result.innerHTML = '<p>An error occurred while processing the video.</p>';
            console.error('Error:', err);
        });
}


// Webcam Snapshot Upload

async function startWebcam() {
    const video = document.getElementById('video');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        document.getElementById('videoContainer').style.display = 'block';
    } catch (err) {
        console.error('Failed to access webcam:', err);
    }
}

function captureFrame() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0);

    canvas.toBlob(blob => {
        if (blob) {
            const formData = new FormData();
            formData.append('file', blob, 'frame.png');
            uploadSnapshot(formData);
        }
    }, 'image/png');
}

function uploadSnapshot(formData) {
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    loading.style.display = 'block';
    result.innerHTML = '';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            loading.style.display = 'none';
            if (data.length === 0) {
                result.innerHTML = '<p>No faces detected in the image.</p>';
            } else {
                let resultText = "<h3>Detected Emotion (Left to Right):</h3>";
                data.forEach((emotion, i) => {
                    resultText += `<p>Person ${i + 1}: Emotion: <strong>${emotion.emotion}</strong>, Confidence: ${emotion.confidence.toFixed(1)}%</p>`;
                });
                resultText += "<p>Note: Persons are ordered from left to right.</p>";
                result.innerHTML = resultText;
            }
        })
        .catch(err => {
            loading.style.display = 'none';
            result.innerHTML = '<p>An error occurred while processing the image.</p>';
            console.error('Error:', err);
        });
}
