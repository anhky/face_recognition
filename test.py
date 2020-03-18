import face_recognition
import cv2
import imutils
from imutils.video import FPS
import numpy as np 
#load camera 
cap = cv2.VideoCapture(0)
fps = FPS().start()
#Load image anđ recognize 
anhky_image = face_recognition.load_image_file("./img/anhky/anhky.jpg")
anhky_face_encoding = face_recognition.face_encodings(anhky_image)[0]
# Create arrays of know face encodings and their names 
known_face_encodings = [
    anhky_face_encoding
]
known_face_names = [
    "anhky"
 
]

while True:
    ret, frame = cap.read()
    rgb_frame = frame [:,:,::-1]
    # Find all faces and faces encodings in the frame 
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    # Loop through each face in this frame ofvideo 
    for (top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
        # See if the face is a match for the knows faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),  font, 1.0, (255, 255,255), 1)

    # Display thw resualting image 
    cv2.imshow('video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break 
    fps.update()
fps.stop()
print("[INFO] elapsed Time : {:.2f}". format(fps.elapsed()))
print("[INFO] appros FPS : {:.2f}". format(fps.fps()))
cap.release()
cv2.destroyAllWindows()
