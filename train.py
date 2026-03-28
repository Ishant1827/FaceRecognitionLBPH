import cv2
import os
import numpy as np

dataset_path = "dataset"
faces = []
labels = []
names = []
label_id = 0

for person in os.listdir(dataset_path):
    names.append(person)
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (200, 200))  # 🔥 FIX HERE
        faces.append(img)
        labels.append(label_id)

    label_id += 1

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))
model.save("trainer.yml")

print("Training Completed Successfully!")
