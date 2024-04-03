from flask import Flask, render_template, Response, request, redirect, send_file,url_for
import os
import cv2
import csv
import face_recognition
import numpy as np
from datetime import datetime
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] =r'D:\att pro\Formal Pictures'
max_recognition_duration = 5
attendance_data = {} 

# Initialize global variables
video_capture = cv2.VideoCapture(0)
encodeListKnown = []
classNames = []
recognized_faces = set()
recognized_face_start_time = {}
marked_names = set()

# Load known face images and encodings
path_to_known_faces = app.config['UPLOAD_FOLDER']
known_face_files = os.listdir(path_to_known_faces)

for cl in known_face_files:
    cur_img = cv2.imread(os.path.join(path_to_known_faces, cl))
    if cur_img is not None:
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(cur_img, num_jitters=3)[0]
        encodeListKnown.append(encode)
        classNames.append(os.path.splitext(cl)[0])

start_time = time.time()
end_time = start_time +attendance_data

# Mark 'In' time for recognized faces
def markInTime(name):
    now = datetime.now()
    date_string = now.strftime('%d-%m-%Y')
    time_string = now.strftime('%H:%M:%S')

    if name not in marked_names:
        marked_names.add(name)
        attendance_data[name] = {"Date": date_string, "In Time": time_string, "Out Time": ""}
        updateCSV(name)

# Mark 'Out' time for recognized faces
def markOutTime(name):
    if name in marked_names and attendance_data[name]["Out Time"] == "":
        now = datetime.now()
        time_string = now.strftime('%H:%M:%S')
        attendance_data[name]["Out Time"] = time_string
        updateCSV(name)

# Update the CSV file with attendance data
def updateCSV(name):
    with open('Attendance.csv', 'w') as f:
        f.write('Name,Date,In Time,Out Time\n')
        for name, data in attendance_data.items():
            f.write(f'{name},{data["Date"]},{data["In Time"]},{data["Out Time"]}\n')

# Read attendance data from the CSV file
def read_attendance_data():
    attendance_data = {}
    with open('Attendance.csv', 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            name = row['Name']
            date = row['Date']
            in_time = row['In Time']
            out_time = row['Out Time']
            if name not in attendance_data:
                attendance_data[name] = []
            attendance_data[name].append({"Date": date, "In Time": in_time, "Out Time": out_time})
    return attendance_data

# Recognize faces in the video feed
def recognize_faces(frame):
    success, frame = video_capture.read()
    frame_small = cv2.resize(frame, (0, 0), None, fx=0.25, fy=0.25)
    frame_small_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    if time.time() <= end_time:
        faces_cur_frame = face_recognition.face_locations(frame_small_rgb, number_of_times_to_upsample=2)
        encodes_cur_frame = face_recognition.face_encodings(frame_small_rgb, faces_cur_frame)

        for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
            matches = face_recognition.compare_faces(encodeListKnown, encode_face, tolerance=0.5)
            face_dis = face_recognition.face_distance(encodeListKnown, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = classNames[match_index].upper()
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if name in recognized_faces:
                    if name in recognized_face_start_time:
                        recognition_duration = time.time() - recognized_face_start_time[name]
                        if recognition_duration >= max_recognition_duration:
                            attendance_message = "Attendance already taken"
                            cv2.putText(frame, attendance_message, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
                            if recognition_duration >= 3600:
                                markOutTime(name)
                        else:
                            text = "Thank you for your attendance"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                            text_x = (frame.shape[1] - text_size[0]) // 2
                            text_y = 30
                            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255, 0), 2)
                            markInTime(name)
                    else:
                        recognized_face_start_time[name] = time.time()
                else:
                    markInTime(name)
                    recognized_faces.add(name)
                    recognized_face_start_time[name] = time.time()
            else:
                unknown_message = "Unknown Person"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, unknown_message, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    return frame

# Video feed generator
def gen():
    while True:
        success, frame = video_capture.read()
        if not success:
            continue

        frame = recognize_faces(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/attendance')
def attendance():
    attendance_data = read_attendance_data()
    return render_template('attendance.html', data=attendance_data)

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

@app.route('/download')
def download_csv():
    # Define the path to the CSV file
    csv_file_path = 'Attendance.csv'

    return send_file(
        csv_file_path,
        as_attachment=True,
        download_name='Attendance.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
