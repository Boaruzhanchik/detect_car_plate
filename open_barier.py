import cv2
import pytesseract
import re
import yolov5

model1 = yolov5.load('yolov5s.pt')
model2 = yolov5.load('yolov5_car_plate_detect.pt')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the allowed plate number
allowed_plate_number = "H961BB82"  # Replace with your allowed plate number

def open_barrier(plate_number):
    # Placeholder for the barrier opening logic
    if any(plate in plate_number for plate in ALLOWED_PLATE_NUMBER):
        print(f"Barrier opened")
    else:
        print(f"Access denied ")
        
#def open_barrier(plate_number):
    # Placeholder for the barrier opening logic
    #if allowed_plate_number in plate_number:
        #print(f"Barrier opened")
    #else:
        #print(f"Access denied ")

for model in [model1, model2]:
    model.conf = 0.25
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = 1000

cap = cv2.VideoCapture('3.mp4')

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
            x, y, w, h = map(int, box)
            roi = frame[y:y+h, x:x+w]
            custom_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            plate_text = pytesseract.image_to_string(roi, config=custom_config)
            
            # Use regular expression to keep only letters and digits
            cleaned_plate_text = re.sub(r'[^a-zA-Z0-9]', '', plate_text)
            
            print(f'Автомобильный номер: {cleaned_plate_text}')
            
            open_barrier(cleaned_plate_text)

    frame = results2.render()[0]
    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
