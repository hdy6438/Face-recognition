import time

import cv2
import dlib
import numpy as np
import setting

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("dlib_model/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_model/dlib_face_recognition_resnet_model_v1.dat")

dataset = np.load("data/dataset.npy")
label = np.load("data/label.npy")

cap = cv2.VideoCapture(0)

begin_time = time.time()
frame_num = 0

while cap.isOpened():
    success,frame = cap.read()
    if success is not True or cv2.waitKey(1) == ord('q'):
        break

    frame = cv2.flip(frame, 1)
    frame_changed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(frame_changed)
    for face in faces:
        frame = cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        shape = sp(frame_changed, face)
        face_descriptor = np.array(facerec.compute_face_descriptor(frame_changed, shape))
        dis = list(np.sqrt(np.sum(np.square(face_descriptor - dataset), 1)))
        name = label[dis.index(min(dis))]
        dis = round(min(dis),4)
        if dis <= setting.min_dis:
            frame = cv2.putText(frame, "{},dis:{}".format(name,dis), (face.left(), face.top()-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        else:
            frame = cv2.putText(frame, "unknown", (face.left(), face.top()-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    frame = cv2.putText(frame, "FACE:{}".format(len(faces)), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    fps = round(frame_num/(time.time()-begin_time),2)
    frame = cv2.putText(frame, "FPS:{}".format(fps), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("cap",frame)
    frame_num +=1


cap.release()
