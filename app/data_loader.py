import os
from PIL import Image
import numpy as np

IMG_SIZE = (128, 128)

def load_images_from_folder(folder_path, categories):
    images = []
    labels = []
    for idx, category in enumerate(categories):
        category_path = os.path.join(folder_path, category)
        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            try:
                img = Image.open(filepath).convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img) / 255.0)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
    return np.array(images), np.array(labels)

def prepare_data(data_path, categories):
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'validation')
    test_path = os.path.join(data_path, 'test')
    
    # Load data for train, validation, and test
    X_train, y_train = load_images_from_folder(train_path, categories)
    X_val, y_val = load_images_from_folder(val_path, categories)
    X_test, y_test = load_images_from_folder(test_path, categories)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
