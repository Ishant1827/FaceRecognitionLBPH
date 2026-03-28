import cv2
import os
import sys
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# -------- RESOURCE PATH FIX (FOR EXE) -------- #
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# -------- LOAD MODEL & DATA -------- #
model = cv2.face.LBPHFaceRecognizer_create()
model.read(resource_path("trainer.yml"))

dataset_path = resource_path("dataset")
names = os.listdir(dataset_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------- CAMERA FIX FOR WINDOWS -------- #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

running = False

# -------- GUI SETUP -------- #
root = tk.Tk()
root.title("AI Face Recognition Dashboard")
root.geometry("1100x650")
root.configure(bg="#0f172a")

# Left Panel
left_panel = tk.Frame(root, bg="#1e293b", width=250)
left_panel.pack(side="left", fill="y")

# Right Panel
right_panel = tk.Frame(root, bg="#0f172a")
right_panel.pack(side="right", expand=True, fill="both")

title = tk.Label(left_panel,
                 text="AI SYSTEM",
                 font=("Arial", 20, "bold"),
                 fg="white",
                 bg="#1e293b")
title.pack(pady=30)

status_label = tk.Label(left_panel,
                        text="Status: Idle",
                        font=("Arial", 14),
                        fg="yellow",
                        bg="#1e293b")
status_label.pack(pady=10)

info_label = tk.Label(left_panel,
                      text="Detected: None",
                      font=("Arial", 12),
                      fg="white",
                      bg="#1e293b")
info_label.pack(pady=10)

video_label = tk.Label(right_panel, bg="black")
video_label.pack(expand=True)

# -------- RECOGNITION LOOP -------- #
def update_frame():
    global running

    if not running:
        return

    ret, frame = cap.read()

    if not ret:
        root.after(10, update_frame)
        return

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
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame,
                    f"{name} ({round(confidence,1)})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

        info_label.config(
            text=f"Detected: {name}\nConfidence: {round(confidence,1)}"
        )

    cv2.putText(frame,
                f"Faces: {face_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

# -------- BUTTON FUNCTIONS -------- #
def start_camera():
    global running
    running = True
    status_label.config(text="Status: Running", fg="lightgreen")
    update_frame()

def stop_camera():
    global running
    running = False
    status_label.config(text="Status: Stopped", fg="red")

def on_closing():
    global running
    running = False
    cap.release()
    root.destroy()

# -------- BUTTONS -------- #
start_btn = tk.Button(left_panel,
                      text="Start Recognition",
                      font=("Arial", 12),
                      bg="#22c55e",
                      fg="white",
                      width=20,
                      command=start_camera)
start_btn.pack(pady=20)

stop_btn = tk.Button(left_panel,
                     text="Stop",
                     font=("Arial", 12),
                     bg="#ef4444",
                     fg="white",
                     width=20,
                     command=stop_camera)
stop_btn.pack(pady=10)

exit_btn = tk.Button(left_panel,
                     text="Exit",
                     font=("Arial", 12),
                     bg="#64748b",
                     fg="white",
                     width=20,
                     command=on_closing)
exit_btn.pack(pady=20)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
