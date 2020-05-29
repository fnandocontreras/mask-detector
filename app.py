import cv2
import tensorflow as tf
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import argparse
from tools import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=None, default=0, help='the video source to analyse, default: 0:webcam')
parser.add_argument('--save', action="store_true", default=False, help='save detected face frames to disk')
parser.add_argument('--rec', action="store_true", default=False, help='record video to disk')
parser.add_argument('--rotate', action="store_true", default=False, help='record video to disk')
args = vars(parser.parse_args())


CLASS_NAMES = ['nomask', 'mask']


#loading models
model = load_model('model.h5')
detector = MTCNN()

cap = cv2.VideoCapture(args['source'])

#recording options
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('record.avi',fourcc, 20.0, (640,480))

def move_next():
    return args['source'] == 0 or cap.isOpened()

while(move_next()):
    # Capture frame-by-frame
    ret, image_np = cap.read()
    if ret==True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #rotate if source is a video file
        if args['source'] != 0 and args['rotate']:
            image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        
        face_boxes = detector.detect_faces(image_np)
        face_boxes, faces = extract_faces(face_boxes, image_np, args)
        if len(faces) > 0:
            preds = model.predict(tf.data.Dataset.from_tensors(faces))

            probas = preds.max(axis=1)
            y_preds = [CLASS_NAMES[c] for c in preds.argmax(axis=1)]
            draw_boxes(image_np, face_boxes, (y_preds, probas))
        
        if args['rec']:
            out.write(image_np)

        cv2.imshow('mask detector', image_np)

    else:
        break
    

    


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()