import cv2
import pytesseract
import sqlite3

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
plate_cascade = cv2.CascadeClassifier(r'C:\Dev\detect_car_number\haarcascade_russian_plate_number.xml')

conn = sqlite3.connect('vehicle_numbers.db')
cursor = conn.cursor()
cursor.execute("SELECT number FROM allowed_numbers")
allowed_plate_numbers = [row[0] for row in cursor.fetchall()]


def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


def main():
    cap = cv2.VideoCapture(r'C:\Dev\detect_car_number\cars\3.mp4')

    while True:
        ret, frame = cap.read()
        frame = enlarge_img(frame, 150)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, frame_thresh = cv2.threshold(
            frame_gray, 127, 255, cv2.THRESH_BINARY)
        plates = plate_cascade.detectMultiScale(
            frame_thresh, scaleFactor=1.1,
            minNeighbors=8,
            minSize=(30, 30))
        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            plate_roi = frame[y:y + h, x:x + w]
            edges = cv2.Canny(frame_gray, 50, 150, apertureSize=3)
            license_plate_text = pytesseract.image_to_string(
                plate_roi,
                config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                lang='rus+eng'
            )

            print(f'Автомобильный номер: {license_plate_text}')
            if license_plate_text.strip() in allowed_plate_numbers:
                print('Проезд разрешен')

            frame_contours = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(frame_contours, contours, -1, (0, 255, 0), 2)
            cv2.putText(frame, f'Car plate: {license_plate_text}',
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 255, 0), 2)

        cv2.imshow('License Plate Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
