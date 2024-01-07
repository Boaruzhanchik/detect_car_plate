
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import itertools

model_path = r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\model1_nomer.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
image_path = r'C:\Dev\Magistratura_Python\itogovyi_proekt\osnovy_mash_ob\3.jpg'
image = cv2.imread(image_path)
resized_image = cv2.resize(image, (input_shape[2], input_shape[1]))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
input_data = np.expand_dims(gray_image, axis=-1)
input_data = np.expand_dims(input_data, axis=0)
input_data = input_data.astype(np.float32) / 255.0

if input_data.shape == tuple(input_shape):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    plt.imshow(gray_image.squeeze(), cmap='gray')
    plt.show()
else:
    print(f"Error: Input shape mismatch. Expected {input_shape}, got {input_data.shape}.")

letters=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']


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


net_out_value = interpreter.get_tensor(output_details[0]['index'])
pred_texts = decode_batch(net_out_value)
pred_texts
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(image_path)

# Convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (128, 64))

# Convert image to float32 and normalize
img = img.astype(np.float32)
img /= 255

img1 = img.T
img1.shape
X_data1 = np.float32(img1.reshape(1, 128, 64, 1))

input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, X_data1)
interpreter.invoke()

net_out_value = interpreter.get_tensor(output_details[0]['index'])
pred_texts = decode_batch(net_out_value)
print(pred_texts)
