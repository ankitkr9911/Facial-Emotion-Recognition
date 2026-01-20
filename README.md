# Facial Emotion Recognition

A real-time facial emotion recognition system using deep learning and computer vision techniques. This project detects human faces in video streams and classifies their emotions into seven categories.

## Features

- Real-time emotion detection from webcam feed
- Face detection using Haar Cascade Classifier
- Deep learning-based emotion classification
- Supports 7 emotion categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

## Prerequisites

- Python 3.x
- OpenCV (cv2)
- TensorFlow/Keras
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ankitkr9911/Facial-Emotion-Recognition.git
cd Facial-Emotion-Recognition
```

2. Install required dependencies:
```bash
pip install opencv-python tensorflow numpy
```

## Project Structure

- `main.py` - Main script for real-time emotion detection from webcam
- `Classification Of Emotions.ipynb` - Jupyter notebook with model training and testing
- `network_emotions.json` - Pre-trained model architecture
- `weights_emotions.hdf5` - Pre-trained model weights
- `haarcascade_frontalface_default.xml` - Haar Cascade classifier for face detection

## Usage

### Real-time Emotion Detection

Run the main script to start real-time emotion detection from your webcam:

```bash
python main.py
```

Press `q` to quit the application.

### Training and Analysis

Open the Jupyter notebook for detailed model training, testing, and analysis:

```bash
jupyter notebook "Classification Of Emotions.ipynb"
```

## How It Works

1. **Face Detection**: The system uses OpenCV's Haar Cascade Classifier to detect faces in each video frame
2. **Preprocessing**: Detected faces are resized to 48x48 pixels and normalized
3. **Emotion Classification**: A pre-trained deep learning model (CNN) predicts the emotion from the processed face image
4. **Display**: The detected emotion is displayed on the video frame in real-time

## Model Details

- Input: 48x48 RGB face images
- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow/Keras
- Output: 7 emotion categories

## License

This project is available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Face detection using OpenCV's Haar Cascade Classifier
- Deep learning framework: TensorFlow/Keras
