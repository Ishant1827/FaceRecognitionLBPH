import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

dataset_path = "dataset"
names = os.listdir(dataset_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

running = False

def start_recognition():
    global running
    running = True
    Thread(target=recognize).start()

def stop_recognition():
    global running
    running = False
    cv2.destroyAllWindows()

def recognize():
    global running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.equalizeHist(roi)
            roi = cv2.resize(roi, (200, 200))

            label, confidence = model.predict(roi)

            if confidence < 70:
                name = names[label]
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({round(confidence,1)})",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)

        cv2.imshow("Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- GUI ---------------- #

root = tk.Tk()
root.title("Face Recognition AI System")
root.geometry("400x250")
root.configure(bg="#1e1e1e")

title = tk.Label(root,
                 text="Face Recognition System",
                 font=("Arial", 16, "bold"),
                 fg="white",
                 bg="#1e1e1e")
title.pack(pady=20)

start_btn = tk.Button(root,
                      text="Start Recognition",
                      font=("Arial", 12),
                      command=start_recognition,
                      bg="green",
                      fg="white",
                      width=20)
start_btn.pack(pady=10)

stop_btn = tk.Button(root,
                     text="Stop",
                     font=("Arial", 12),
                     command=stop_recognition,
                     bg="red",
                     fg="white",
                     width=20)
stop_btn.pack(pady=10)

exit_btn = tk.Button(root,
                     text="Exit",
                     font=("Arial", 12),
                     command=root.quit,
                     bg="gray",
                     fg="white",
                     width=20)
exit_btn.pack(pady=10)

root.mainloop()
