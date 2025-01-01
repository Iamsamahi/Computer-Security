from sklearn.model_selection import train_test_split
import shutil
import os

def split_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in the directory
    files = [imageFiles for imageFiles in os.listdir(input_dir) if imageFiles.endswith(('.jpg', '.jpeg', '.png'))]
    
    # First, split into 80% training and 20% validation+test
    train, temp = train_test_split(files, test_size=0.2, random_state=42)
    
    # Split the 20% into 15% test and 5% validation
    val, test = train_test_split(temp, test_size=0.75, random_state=42)  # 0.75 of 20% -> 15% for test and 5% for validation

    # Save the files into corresponding directories
    for split, split_files in zip(['train', 'val', 'test'], [train, val, test]):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for file in split_files:
            shutil.copy(os.path.join(input_dir, file), os.path.join(split_dir, file))

split_data("Processed_Dataset/Live/cropped_frames", "Dataset_Splits/Live")
split_data("Processed_Dataset/Spoof/cropped_frames", "Dataset_Splits/Spoof")
