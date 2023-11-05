# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:21:20 2023

@author: neeltron
"""

import cv2
import numpy as np
import face_recognition

ref = face_recognition.load_image_file('static/patient0.jpg')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

face_locations = face_recognition.face_locations(ref)
if not face_locations:
    print("no faces ")
else:
    train_encode = face_recognition.face_encodings(ref, known_face_locations=face_locations)[0]

test = face_recognition.load_image_file('static/patienttest.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

test_encode = face_recognition.face_encodings(test)
if not test_encode:
    print("no faces")
else:
    result = face_recognition.compare_faces([train_encode], test_encode[0])
    print("res", result)

if face_locations:
    top, right, bottom, left = face_locations[0]
    cv2.rectangle(ref, (left, top), (right, bottom), (255, 0, 255), 1)

cv2.imshow('ref', ref)
cv2.waitKey(0)