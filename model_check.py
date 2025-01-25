import onnx
import onnxruntime as ort
import numpy as np
import cv2


model_path = "chaesiktak_weights.onnx"
image_path = "test.jpg"
class_names_path = "chaesiktak_classes.txt"

def load_class_names(class_file):
    with open(class_file, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

class_names = load_class_names(class_names_path)

try:
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    print("ONNX model 추론 가능")
except Exception as e:
    print(f"ONNX model 추론 안됨: {e}")
    exit()


def preprocess_image(image_path, input_size=640):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 이미지 배치 차원 추가, 한 장씩 입력 가정.
    return image

input_tensor = preprocess_image(image_path)


session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

outputs = session.run([output_name], {input_name: input_tensor})

detections = outputs[0][0] 
confidence_threshold = 0.5 


print("감지 재료:")
for detection in detections:
    confidence = detection[4]
    if confidence > confidence_threshold:
        x, y, w, h = detection[:4]
        class_probs = detection[5:5 + len(class_names)]
        class_id = np.argmax(class_probs)  # 가장 높은 신뢰도 가진 값.
        class_label = class_names[class_id]

        print(f"Class: {class_label} (ID: {class_id}), Confidence: {confidence:.2f}")