import numpy as np
from PIL import Image

def predict_image(model, image_path, categories, img_size=(128, 128)):
    try:
        img = Image.open(image_path).convert('RGB').resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        return predicted_class
    except Exception as e:
        return f"Error: {e}"
