import cv2
import dlib
import numpy as np

import setting

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("dlib_model/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_model/dlib_face_recognition_resnet_model_v1.dat")
cap = cv2.VideoCapture(0)

face_count = 0
datas = []
while cap.isOpened():
    success,frame = cap.read()
    if success is not True or cv2.waitKey(1) == ord('q') or face_count == setting.need_frame_num:
        break

    frame = cv2.flip(frame, 1)
    frame_changed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = detector(frame_changed)
    if len(face) ==1:
        face = face[0]
        frame = cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        shape = sp(frame_changed,face)
        face_descriptor = np.array(facerec.compute_face_descriptor(frame_changed, shape))
        datas.append(face_descriptor)
        print(face_descriptor)
        face_count +=1

    frame = cv2.putText(frame, "face_count:{}".format(face_count), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("cap",frame)

cap.release()
cv2.destroyAllWindows()

datas = np.array(datas)
datas = np.mean(datas,axis=0)
name = input("input name ==>")
np.save("data/faces/{}".format(name),datas)

