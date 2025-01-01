import os
import cv2

def preprocess_live_images(input_dir, output_dir, target_size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, target_size)  # Resize
            img_normalized = img_resized / 255.0       # Normalize
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, (img_normalized * 255).astype('uint8'))  # Save preprocessed image

preprocess_live_images("Dataset/Live/images", "Processed_Dataset/Live/images")

def extract_frames(input_dir, output_dir, frame_rate=10):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            video_capture = cv2.VideoCapture(video_path)
            success, frame_number = True, 0

            while success:
                success, frame = video_capture.read()
                if success and frame_number % int(video_capture.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
                    frame_filename = os.path.join(output_dir, f"{filename}_frame_{frame_number}.jpg")
                    cv2.imwrite(frame_filename, frame)
                frame_number += 1


extract_frames("Dataset/Live/videos", "Processed_Dataset/Live/frames")
extract_frames("Dataset/Spoof/videos", "Processed_Dataset/Spoof/frames")



def crop_faces(input_dir, output_dir, face_cascade_path="haarcascades\\haarcascade_frontalface_default.xml"):
    # Create a CascadeClassifier object
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Check if the Haarcascade file was loaded successfully
    if face_cascade.empty():
        raise IOError("Haarcascade file not found or could not be loaded.")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image file extensions
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # Crop and save each detected face
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, face)

# Call the function for different datasets
crop_faces("Processed_Dataset/Live/frames", "Processed_Dataset/Live/cropped_frames")
crop_faces("Processed_Dataset/Spoof/frames", "Processed_Dataset/Spoof/cropped_frames")
