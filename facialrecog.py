import face_recognition as fr
import cv2
import numpy as np 
import csv 
import os 
from datetime import datetime



video_capture = cv2.VideoCapture(0)

path = 'C:/Users/mekal/OneDrive/Desktop/FacialRecog/images/'

known_face_encoding = []
known_faces_names = []

img_path = os.listdir(path)
for x in img_path:
    x_image = fr.load_image_file(path + x)
    imgs_path = path + x
    x_encoding = fr.face_encodings(x_image)[0] 
    
    known_face_encoding.append(x_encoding)
    known_faces_names.append(os.path.splitext(os.path.basename(imgs_path)) [0].capitalize())
    
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
 
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open("Attendence of "+current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame,face_locations)
        face_names = []  
        for face in face_encodings:
            matches = fr.compare_faces(known_face_encoding,face)
            name="" 
            face_distance = fr.face_distance(known_face_encoding,face)
            best_match_index = np.argmin(face_distance) 
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_date = now.strftime("%Y-%m-%d")
                    current_time = now.strftime("%H:%M")
                    lnwriter.writerow([name,current_date,current_time])
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
video_capture.release()
cv2.destroyAllWindows()
f.close()
