import cv2
import os
import numpy as np

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

dataset_path = "dataset"
names = os.listdir(dataset_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count = len(faces)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.equalizeHist(roi)
        roi = cv2.resize(roi, (200, 200))

        label, confidence = model.predict(roi)

        if confidence < 70:
            name = names[label]
            color = (0, 255, 0)  # Green
        else:
            name = "Unknown"
            color = (0, 0, 255)  # Red

        # Fancy rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Background label box
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)

        # Name + confidence
        cv2.putText(frame,
                    f"{name}  {round(confidence,1)}",
                    (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

    # Top title
    cv2.putText(frame,
                "AI Face Recognition System",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    # Face counter
    cv2.putText(frame,
                f"Faces Detected: {face_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.imshow("AI Face Recognition - Ishant Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
