import cv2
from ultralytics import YOLO
import winsound
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from twilio.rest import Client
from dotenv import load_dotenv
import os
load_dotenv()
load_dotenv(dotenv_path=".env")

# -----------------------
# LOAD MODEL
# -----------------------
model = YOLO("best_small.pt")

account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_TOKEN")

sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
email_password = os.getenv("EMAIL_PASS")

twilio_number = os.getenv("TWILIO_NUMBER")
your_number = os.getenv("YOUR_NUMBER")


# -----------------------
# FUNCTIONS
# -----------------------

def sound_alarm():
    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)

def send_email(image_path):

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "🔥 Fire Detected Alert"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        text = MIMEText("Fire detected! Please check immediately.")
        msg.attach(text)

        with open(image_path, "rb") as f:
            img = MIMEImage(f.read())
            msg.attach(img)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, email_password)
        server.send_message(msg)
        server.quit()

        print("Email alert sent")

    except Exception as e:
        print("Email failed:", e)

def send_sms():

    try:
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body="🔥 Fire detected! Check immediately!",
            from_=twilio_number,
            to=your_number
        )

        print("SMS sent:", message.sid)

    except Exception as e:
        print("SMS failed:", e)

# -----------------------
# CAMERA
# -----------------------

cap = cv2.VideoCapture(0)

alarm_on = False
fire_frames = 0
last_alert_time = 0

ALERT_COOLDOWN = 30  # seconds

# -----------------------
# MAIN LOOP
# -----------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45)

    fire_detected = False

    for box in results[0].boxes:

        confidence = float(box.conf)
        cls = int(box.cls[0])
        label = model.names[cls]

        if confidence < 0.55:
            continue

        if label.lower() == "fire":

            fire_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

            cv2.putText(frame,f"Fire {confidence:.2f}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,0,255),2)

    # -----------------------
    # STABILITY CHECK
    # -----------------------

    if fire_detected:
        fire_frames += 1
    else:
        fire_frames = 0
        alarm_on = False

    # -----------------------
    # ALERT TRIGGER
    # -----------------------

    if fire_frames >= 6:

        cv2.putText(frame,"FIRE DETECTED!",
                    (40,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0,0,255),
                    3)

        if not alarm_on:

            alarm_on = True
            threading.Thread(target=sound_alarm).start()

        current_time = time.time()

        if current_time - last_alert_time > ALERT_COOLDOWN:

            last_alert_time = current_time

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = f"fire_{timestamp}.jpg"

            cv2.imwrite(image_path, frame)

            print("Image saved:", image_path)

            threading.Thread(target=send_email, args=(image_path,)).start()
            threading.Thread(target=send_sms).start()

    # -----------------------
    # DISPLAY
    # -----------------------

    cv2.imshow("Fire and Smoke Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()