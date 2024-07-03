import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog

# Paths to training data folders
empty_folder_path = "parking/clf-data/empty"
occupied_folder_path = "parking/clf-data/not_empty"
video_path = "parking/parking_1920_1080.mp4"
mask_path = "parking\mask_1920_1080.png"


# Read the mask image
mask = cv2.imread(mask_path, 0)

# Capture video
cap = cv2.VideoCapture(video_path)

myvideo = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count < 10:
        frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))  # Resizing to match mask
        myvideo.append(frame)
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

# Function to load images and assign labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize for uniformity
            images.append(img)
            labels.append(label)
    return images, labels

# Load images
empty_images, empty_labels = load_images_from_folder(empty_folder_path, 0)
occupied_images, occupied_labels = load_images_from_folder(occupied_folder_path, 1)

# Concatenate and shuffle data
X_data = np.concatenate((empty_images, occupied_images), axis=0)
y_data = np.concatenate((empty_labels, occupied_labels), axis=0)

# Normalize pixel values
X = X_data.astype('float32') / 255.0

# One-hot encode labels
y = to_categorical(y_data, num_classes=2)

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {accuracy}")

# Function to classify parking spots in a frame
def classify_parking_spots(frame, parking_spots, model):
    spot_images = [frame[y:y+h, x:x+w] for (x, y, w, h) in parking_spots]
    spot_images_resized = [cv2.resize(spot, (64, 64)) for spot in spot_images]
    spot_images_resized = np.array(spot_images_resized).astype('float32') / 255.0
    spot_predictions = model.predict(spot_images_resized)
    return spot_predictions

# Find contours in the mask image
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract bounding boxes for each parking spot
parking_spots = [cv2.boundingRect(contour) for contour in contours]

# Classify parking spots in each frame
for i, frame in enumerate(myvideo):
    spot_predictions = classify_parking_spots(frame, parking_spots, model)
    occupied_spots = np.sum(spot_predictions[:, 1] > 0.5)
    empty_spots = len(parking_spots) - occupied_spots
    print(f"Frame {i+1}: {occupied_spots} occupied spots, {empty_spots} empty spots")

# Calculate evaluation metrics
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1}")

# Get video properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up video writer
output_video_path =("YOUR OUTPUT PATH")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the entire video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the frame is resized to match the mask
    frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))

    spot_predictions = classify_parking_spots(frame, parking_spots, model)
    occupied_spots = np.sum(spot_predictions[:, 1] > 0.5)
    empty_spots = len(parking_spots) - occupied_spots

    # Annotate the frame
    frame_copy = frame.copy()
    for idx, (x, y, w, h) in enumerate(parking_spots):
        if spot_predictions[idx, 1] > 0.5:  # occupied
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h),  (0, 0, 255), 2)  # red
        else:  # empty
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h),(0, 255, 0), 2)  # green

    cv2.putText(frame_copy, f'Occupied: {occupied_spots}, Empty: {empty_spots}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    out.write(frame_copy)

# Release the video writer and video capture
cap.release()
out.release()

print(f"Output video saved at {output_video_path}")

