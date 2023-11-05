# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 16:21:20 2023

@author: neeltron
"""

import cv2
import numpy as np
import face_recognition
from flask import Flask, render_template

dict = {0: ["neel", "5’10”", "170 lbs", "Ligma", "Ligma"],
        1: ["b", "5’5”", "160 lbs", "Congenital Heart Disease", "20 mgs Lisinoprol, 10 mgs Mannitol, 15 mgs Aprostadil"],
        2: ["c", "5’3”", "150 lbs", "Lupus", "5 mgs Hydroxychloroquine, 10 mgs Prednisone, 50 mgs Ibuprofen"],
        3: ["d", "6’0”", "210 lbs", "ADHD", "5 mgs Adderall twice a day"]
        }

def recognize(ref, test):
    ref = face_recognition.load_image_file('static/'+ref)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(ref)
    if not face_locations:
        print("no ref faces")
    else:
        train_encode = face_recognition.face_encodings(ref, known_face_locations=face_locations)[0]
    test = face_recognition.load_image_file('static/' + test)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test_encode = face_recognition.face_encodings(test)
    if not test_encode:
        print("no test faces")
    else:
        result = face_recognition.compare_faces([train_encode], test_encode[0])
        print("res", result)
    
    if face_locations:
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(ref, (left, top), (right, bottom), (255, 0, 255), 1)
    
    cv2.imshow('ref', ref)
    cv2.waitKey(0)

app = Flask('app', static_folder='static', template_folder='templates')

@app.route('/')
def index():
  return render_template('index.html', data = dict)

@app.route('/capture')
def capture():
    return render_template("capture.html")

app.run(host='0.0.0.0', port=8080)
