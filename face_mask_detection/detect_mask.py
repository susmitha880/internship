import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

model = load_model("models/mask_detector.h5")

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100,100))
        face = np.reshape(face, (1,100,100,3)) / 255.0

        pred = model.predict(face)
        label = le.classes_[np.argmax(pred)]

        color = (0,255,0) if label == "with_mask" else (0,0,255)

        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()