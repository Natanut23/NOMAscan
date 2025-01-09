import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# โหลดโมเดล
model_path = r'D:\Work-24 it\MelanomaSenser.Project\c-CODE-VS\NMs-PJ.vscode\model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

model = load_model(model_path)

# หมวดหมู่ที่ใช้ (ต้องตรงกับที่ใช้ตอนเทรนโมเดล)
categories = ['basal_cell_carcinoma', 'benign_keratosis_lesions', 'melanocytic_nevi', 'melanoma']

# แก้ไขฟังก์ชัน predict_single_image
def predict_single_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}. Please check the path.")
    
    # โหลดภาพและแปลงเป็น array
    img = load_img(image_path, target_size=(128, 128))  # ปรับขนาดภาพให้ตรงกับที่โมเดลต้องการ
    img_array = img_to_array(img) / 255.0  # ทำ normalization
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติสำหรับ batch
    
    # ทำนายผล
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    return categories[predicted_class], confidence

    
    return categories[predicted_class], confidence

if __name__ == "__main__":
    image_path = input("ใส่ path ของภาพที่ต้องการทดสอบ: ")
    predicted_label, confidence = predict_single_image(image_path)
    print(f"ผลการพยากรณ์: {predicted_label} (ความมั่นใจ: {confidence:.2f})")
