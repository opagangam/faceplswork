# utils.py

import os
import cv2
import face_recognition
import numpy as np
import mediapipe as mp

#setting up FaceMesh to use  for liveliness
mesh_model = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Image loader 
def get_img(path):
    img = cv2.imread(path)
    return img

# Using face_recognition lib to find people in image
def find_faces_img(img):
    boxes = face_recognition.face_locations(img)
    return boxes

# Repeating here because sometimes video frame detection does not work properly
def find_faces_frame(f):
    return face_recognition.face_locations(f)

# Crude check to see if someone’s alive in the frame
def is_real_person(full_frame):
    # convert to RGB bc Mediapipe needs it
    rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)

    try:
        res = mesh_model.process(rgb)
        if res.multi_face_landmarks:
            return True
    except Exception as err:
        print("FaceMesh failed:", err)

    return False

# Basic video analysis: count faces and whether they blink


import cv2
import face_recognition

def analyze_vid(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1, -1  # Using -1 to indicate failure

    known_faces = []
    total_new_faces = 0
    real_humans = 0

    frame_index = 0  # For debugging if needed

    while True:
        success, frame = cap.read()
        if not success:
            # print("End of video or read error at frame", frame_index)
            break

        frame_index += 1
        # Could resize for speed if needed, but leaving full size for now

        locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, locations)

        for idx, (enc, (top, right, bottom, left)) in enumerate(zip(encodings, locations)):
            matches = face_recognition.compare_faces(known_faces, enc, tolerance=0.6)
            if not any(matches):
                # Definitely haven't seen this one before
                known_faces.append(enc)
                total_new_faces += 1

                face_img = frame[top:bottom, left:right]
                try:
                    if is_real_person(face_img):
                        real_humans += 1
                except Exception as e:
                    print(f"Error in liveliness check at frame {frame_index}, face #{idx}: {e}")
                    continue  # move on to the next face

    cap.release()
    return total_new_faces, real_humans

