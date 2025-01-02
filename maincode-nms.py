from app.data_loader import prepare_data
from app.model import build_model

# เตรียมข้อมูล
data_path = r'D:\Work-24 it\MelanomaSenser.Project\b-Skin-Disease.Data\DataRTU'  # เส้นทางโฟลเดอร์ข้อมูล
categories = ['basal_cell_carcinoma', 'benign_keratosis_lesions', 'melanocytic_nevi', 'melanoma']

# Load train, validation, and test datasets
X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_path, categories)

# สร้างโมเดล
model = build_model(input_shape=(128, 128, 3), num_classes=len(categories))

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.2f}")

model.save('model.h5')
print("Model saved!")
