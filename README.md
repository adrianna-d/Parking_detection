
# Parking Spot Detection and Classification

This project utilizes computer vision techniques and deep learning to detect and classify free parking spots in a given video.

## Overview

The system processes a video feed of a parking lot, identifying whether each parking spot is occupied or vacant. It employs a Convolutional Neural Network (CNN) trained on labeled images of parking spots to make predictions in real-time.

## Features

- **Video Input:** Supports any video file where parking spots need to be analyzed.
- **Real-time Classification:** Each frame of the video is analyzed to determine the status of each parking spot.
- **Output Video Generation:** An annotated video is generated showing identified occupied and vacant spots.
- **Evaluation Metrics:** Provides metrics such as accuracy, precision, recall, and F1 score to evaluate the model's performance.

## Installation

### Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- TensorFlow (`pip install tensorflow`)
- Scikit-image (`pip install scikit-image`)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/parking_project.git
   cd parking_project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download or configure the video and mask file paths in your code (`video_path` and `mask_path` variables).

## Usage

1. Ensure the video and mask are correctly set up.
2. Run the main script:

   ```bash
   python parking_spot_detection.py
   ```

3. The program will process the video, classify parking spots, and generate an annotated output video.

## Model Training

- The CNN model is trained using images from two directories:
  - `empty_folder_path`: Images of empty parking spots.
  - `occupied_folder_path`: Images of occupied parking spots.

- The model architecture includes convolutional layers with max-pooling, fully connected layers, and dropout to prevent overfitting.

## Files

- **parking_spot_detection.py:** Main script for detecting and classifying parking spots.
- **requirements.txt:** List of Python dependencies.
- **parking_1920_1080.mp4:** Example input video
- **mask_1920_1080** - Mask for the project
- **project_presentation** - .pdf of what was done for the project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was developed as part of IronHack bootcamp project on computer vision.
- @sofiaggoncalves is a co-author of the project.

