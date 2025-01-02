from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
from keras.models import load_model
from app.predict import predict_image
import sys
import os

# ตั้งค่า Python Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
line_bot_api = LineBotApi('2006735158')  # เปลี่ยนเป็น Channel Access Token จริง
handler = WebhookHandler('13a8ee9fdc8259b37236ad5510c103f4')         # เปลี่ยนเป็น Channel Secret จริง

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    handler.handle(body, signature)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    reply = "กรุณาส่งรูปภาพเพื่อตรวจสอบ หรือพิมพ์ 'predict' เพื่อเริ่มการทำงาน"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # ดึงภาพและพยากรณ์
        message_content = line_bot_api.get_message_content(event.message.id)
        temp_file = 'temp.jpg'

        # บันทึกภาพชั่วคราว
        with open(temp_file, 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)

        # โหลดโมเดล
        model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
        model = load_model(model_path)

        # หมวดหมู่ของผลลัพธ์
        categories = ['basal_cell_carcinoma', 'benign_keratosis_lesions', 'melanocytic_nevi', 'melanoma']

        # พยากรณ์ผลลัพธ์
        result = predict_image(model, temp_file, categories)

        # สร้างข้อความตอบกลับ
        reply = f"ผลการทำนาย: {result}"
    except Exception as e:
        # จัดการข้อผิดพลาด
        reply = f"เกิดข้อผิดพลาด โปรดติดต่อ DEV: {str(e)}"
    finally:
        # ลบไฟล์ชั่วคราว
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # ตอบกลับไปยังผู้ใช้
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
