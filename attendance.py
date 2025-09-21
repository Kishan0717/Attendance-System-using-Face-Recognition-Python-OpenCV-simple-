import cv2
import face_recognition
import numpy as np
import datetime

# Load sample image & encode
known_image = face_recognition.load_image_file("data/user1.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

known_faces = [known_encoding]
known_names = ["User1"]

# Open webcam
video_capture = cv2.VideoCapture(0)

attendance_list = []

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            
            if name not in attendance_list:
                now = datetime.datetime.now()
                attendance_list.append(name)
                print(f"{name} marked present at {now.strftime('%H:%M:%S')}")

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
