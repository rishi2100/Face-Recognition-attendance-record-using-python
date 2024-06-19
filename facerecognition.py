import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

rishi_image = face_recognition.load_image_file("rishi.jpg")
rishi_encoding = face_recognition.face_encodings(rishi_image)[0]

priya_image = face_recognition.load_image_file("priya.jpg")
priya_encoding = face_recognition.face_encodings(priya_image)[0]

rekha_image = face_recognition.load_image_file("rekha.jpg")
rekha_encoding = face_recognition.face_encodings(rekha_image)[0]

venky_image = face_recognition.load_image_file("venky.png")
venky_encoding = face_recognition.face_encodings(venky_image)[0]

known_face_encoding = [rishi_encoding, priya_encoding, rekha_encoding, venky_encoding]
known_face_names = ["Rishi", "Priya", "Rekha", "Venky"]
employees = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime('%Y-%m-%d')
f = open(f"{current_date}.csv", "w", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                
                if name in employees:
                    employees.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    
    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()