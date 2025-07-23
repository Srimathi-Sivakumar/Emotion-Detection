# Emotion-Detection
This to detect the emotion using open cv and Deepface 
!pip install deepface
!pip install opencv-python-headless

# Import required libraries
import cv2
from deepface import DeepFace
import os
from google.colab import files

# Upload the video files
uploaded = files.upload()

# Get the uploaded video file name
video_path = list(uploaded.keys())[0]

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

# Check if video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    # Get the video frame dimensions (width and height)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object to save the output video
    output_filename = 'emotion_detection_output_deepface.mp4'
    out = cv2.VideoWriter(output_filename,
                          cv2.VideoWriter_fourcc(*'MP4V'),
                          10,  # FPS (frames per second)
                          (frame_width, frame_height))

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        try:
            # Detect emotions using DeepFace
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # If multiple faces are detected, DeepFace will return a list
            if isinstance(analysis, list):
                for result in analysis:
                    # Extract the bounding box coordinates and dominant emotion
                    dominant_emotion = result['dominant_emotion']
                    box = result['region']  # Bounding box region (x, y, width, height)
                    x, y, w, h = box['x'], box['y'], box['w'], box['h']

                    # Annotate the frame with the dominant emotion
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # If only one face is detected, extract the single result
                dominant_emotion = analysis['dominant_emotion']
                box = analysis['region']  # Bounding box region (x, y, width, height)
                x, y, w, h = box['x'], box['y'], box['w'], box['h']

                # Annotate the frame with the dominant emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error analyzing frame: {e}")

        # Write the annotated frame to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    # Close any OpenCV windows
    cv2.destroyAllWindows()

    # Check if the output file was created and download it
    if os.path.exists(output_filename):
        files.download(output_filename)
    else:
        print("Error: The output file was not created.")
