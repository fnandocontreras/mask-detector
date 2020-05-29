import tensorflow as tf
import numpy as np
import cv2
import os
import uuid



IMG_HEIGHT = 160
IMG_WIDTH = 160

def decode_img(img):
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def extract_faces(result_list, img, args):
    # plot each face as a subplot
    faces = []
    boxes = []
    for i in range(len(result_list)):
        img_width, img_height, img_channels = img.shape
        
        x1, y1, width, height = result_list[i]['box']
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = width + x1
        y2 = height + y1

        box = (x1, y1, x2, y2)
        crop = img[y1:y2, x1:x2, :]
        if crop.shape[0] * crop.shape[1] != 0:
            decoded_face_img = decode_img(crop)
            faces.append(decoded_face_img)
            boxes.append(result_list[i])

            #export detected faces as images to disk
            if args['save']:
                unique_filename = str(uuid.uuid4())
                filename = f'{unique_filename}.jpg'
                cv2.imwrite(os.path.join('export/', filename), crop)
                cv2.imshow('zoom', crop)

    return boxes, faces


def draw_boxes(image, result_list, predictions):
    preds, probas = predictions
    for i in range(len(result_list)):
        result = result_list[i]
        x, y, width, height = result['box']
        color=(0,0,255)#red in BGR
        pred_class = preds[i]
        if pred_class == 'mask':
            color = (0,255,0)#green in BGR
        p1 = (x, y) 
        p2 = (x + width, y + height) 
        thickness = 2
        cv2.rectangle(image, p1, p2, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        score = np.round(probas[i], 2)
        label = f'{pred_class} : {str(score)}'
        cv2.putText(image,label,(x,y-8), font, 0.5,color,2,cv2.LINE_AA)