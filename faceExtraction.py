import cv2
import os

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the vidoe file
video_capture = cv2.VideoCapture('faces2.mp4')

# Define the directory to save the extracted faces
output_dir = 'faces/'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


frame_count = 0
skip_frames = 5  # we are going to skep every 5 frames.
# Otherwise we end up having the almost same picture many times as the frames contain almost identical faces

# Loop over the video frames from the video
while True:
    # Read the next frame from the video
    ret, frame = video_capture.read()

    # Exit if the video has ended
    if not ret:
        break

     # Skip frames if necessary
    if frame_count % skip_frames != 0:
        frame_count += 1
        continue

    # Conver the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over the detected faces and crop them from the frame
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Display the cropped face image (for testing puerposes)
        cv2.imshow('Face', face)
        cv2.waitKey(1)
        face_filename = os.path.join(output_dir, f"face_{frame_count}.jpg")
        cv2.imwrite(face_filename, face)

    frame_count += 1

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
