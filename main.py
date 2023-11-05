# -*- coding: utf-8 -*-
"""
Created on Sat Nov 4 16:21:20 2023
@author: neeltron
"""

import cv2
import face_recognition
from flask import Flask, render_template, jsonify, request, redirect, session
import base64

data_flow = []
flag = 0
final_data = ["", "", "", "", ""]

data_dict = {
    0: ["neel", "10’10”", "170 lbs", "Ligma", "Ligma"],
    1: ["ariana", "5’5”", "160 lbs", "Congenital Heart Disease", "20 mgs Lisinoprol, 10 mgs Mannitol, 15 mgs Aprostadil"],
    2: ["victor", "5’5”", "150 lbs", "Fever", "5 mgs Hydroxychloroquine, 10 mgs Prednisone, 50 mgs Ibuprofen"],
    3: ["jawad", "6’0”", "210 lbs", "ADHD", "5 mgs Adderall twice a day"]
}

def recognize(ref, test):
    ref = face_recognition.load_image_file('static/' + ref)
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(ref)
    if not face_locations:
        print("no ref faces")
        return None
    train_encode = face_recognition.face_encodings(ref, known_face_locations=face_locations)[0]
    
    test = face_recognition.load_image_file('static/' + test)
    test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
    test_encode = face_recognition.face_encodings(test)
    if not test_encode:
        print("no test faces")
        return None
    result = face_recognition.compare_faces([train_encode], test_encode[0])
    print("res", result)
    return result[0]

app = Flask('app', static_folder='static', template_folder='templates')
app.secret_key = 'testlol'

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/patient')
def patient():
    return render_template('index.html', data=data_dict)

@app.route('/capture')
def capture():
    if flag == 0:
        return render_template("capture.html", values=final_data)
    else:
        return render_template("capture.html", values=final_data)

@app.route('/save_photo', methods=['POST'])
def save_photo():
    try:
        global final_data, flag
        flag = 1
        final_data = ["", "", "", "", ""]
        data = request.get_json()
        photo_data = data.get('photoData')

        file_name = 'static/test.jpg'

        with open(file_name, 'wb') as f:
            f.write(base64.b64decode(photo_data.split(',')[1]))

        ref_image = 'test.jpg'
        for key, values in data_dict.items():
            name, _, _, _, _ = values
            test_image = f'{name}.jpg'
            result = recognize(ref_image, test_image)

            if result == True:
                final_data = data_dict[key]
                data_flow.append(name)
                break

        return render_template('capture.html', values=final_data)

    except Exception as e:
        return render_template('capture.html', values=final_data)

if __name__ == '__main__':
    app.run(ssl_context='adhoc', host='0.0.0.0', port=8080)
