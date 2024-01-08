import cv2
#import pytesseract
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yolov5
import itertools
import ALLOWED_PLATE_NUMBER

model_path = r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\model1_nomer.tflite'
model1 = yolov5.load(r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\yolov5s.pt')
model2 = yolov5.load(r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\yolov5_car_plate_detect_v2 (2).pt')


interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']



allowed_plate_number = "H961BB82"

letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


def open_barrier(plate_number):
    if any(plate in plate_number for plate in ALLOWED_PLATE_NUMBER):
        print(f"Barrier opened")
    else:
        print(f"Access denied ")

for model in [model1, model2]:
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

cap = cv2.VideoCapture(r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\1.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results1 = model1(frame)
    results2 = model2(frame)

    predictions2 = results2.pred[0]
    boxes2 = predictions2[:, :4]
    scores2 = predictions2[:, 4]
    categories2 = predictions2[:, 5]

    for box, score, category in zip(boxes2, scores2, categories2):
        if score > model2.conf:
            x_min, y_min, x_max, y_max = map(int, box)
            roi = frame[y_min:y_max, x_min:x_max]
            cv2.imshow('roi',roi)
            resized_image = cv2.resize(roi, (input_shape[2], input_shape[1]))
            cv2.imshow('resized_image',resized_image)
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            input_data = np.expand_dims(gray_image, axis=-1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data.astype(np.float32) / 255.0

    
            if input_data.shape == tuple(input_shape):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                net_out_value = interpreter.get_tensor(output_details[0]['index'])
                pred_texts = decode_batch(net_out_value)
                cleaned_plate_text = pred_texts[0]

                print(f'License Plate: {cleaned_plate_text}')

                cleaned_plate_text = re.sub(r'[^a-zA-Z0-9]', '', cleaned_plate_text)

                #open_barrier(cleaned_plate_text)

    frame = results2.render()[0]
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
