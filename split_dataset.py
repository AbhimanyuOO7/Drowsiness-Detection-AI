import os
import shutil
import random

def split_dataset(source_dir, train_dir, valid_dir, test_size=0.2):
    """
    Splits the dataset into training and validation sets.
    
    Args:
    source_dir (str): Path to the original dataset folder containing subfolders of each class.
    train_dir (str): Path where training data will be stored.
    valid_dir (str): Path where validation data will be stored.
    test_size (float): Proportion of the dataset to include in the validation set (0.2 = 20%).
    """
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    for category in os.listdir(source_dir):  # Loop through each category (e.g., 'Closed_Eyes', 'Open_Eyes')
        category_path = os.path.join(source_dir, category)
        
        if not os.path.isdir(category_path):
            continue
        
        images = os.listdir(category_path)
        random.shuffle(images)
        
        split_index = int(len(images) * (1 - test_size))
        train_images = images[:split_index]
        valid_images = images[split_index:]

        # Create category folders under train_dir and valid_dir
        train_category_path = os.path.join(train_dir, category)
        valid_category_path = os.path.join(valid_dir, category)
        
        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(valid_category_path, exist_ok=True)

        # Copy the training images
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_category_path, img))

        # Copy the validation images
        for img in valid_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(valid_category_path, img))

        print(f"Category '{category}' - Training images: {len(train_images)}, Validation images: {len(valid_images)}")


# Paths to your dataset
source_dir = 'C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/datasets'
train_dir = 'C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/data/train'
valid_dir = 'C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/data/valid'

# Splitting the dataset
split_dataset(source_dir, train_dir, valid_dir)
