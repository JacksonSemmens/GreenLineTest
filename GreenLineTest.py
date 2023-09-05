# image_viewer_with_face_detection.py

import io
import os
import PySimpleGUI as sg
import cv2
from PIL import Image
import numpy as np

file_types = [("Pictures", "*.jpg"),
              ("All files (*.*)", "*.*")]

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image

def main():
    layout = [
        [sg.Image(key="-IMAGE-")],
        [
            sg.Text("Image File"),
            sg.Input(size=(25, 1), key="-FILE-"),
            sg.FileBrowse(file_types=file_types),
            sg.Button("Load Image"),
        ],
    ]

    window = sg.Window("Image Viewer with Face Detection", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Load Image":
            filename = values["-FILE-"]
            if os.path.exists(filename):
                image = cv2.imread(filename)
                image_with_faces = detect_faces(image)  # Detect faces using the integration
                image_with_faces = Image.fromarray(image_with_faces)
                image_with_faces.thumbnail((400, 400))
                bio = io.BytesIO()
                image_with_faces.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())

    window.close()

if __name__ == "__main__":
    main()
