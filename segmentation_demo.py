import cv2
import onnxruntime as ort
import numpy as np
import requests
from collections import Counter
import time


model_path = "chaesiktak_weights.onnx"
class_names_path = "chaesiktak_classes.txt"


def load_class_names(class_file):
    with open(class_file, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

class_names = load_class_names(class_names_path)


def preprocess_image(image, input_size=640):
    image_resized = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 배치 추가
    return image


def detect_objects(image):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_tensor = preprocess_image(image)
    outputs = session.run([output_name], {input_name: input_tensor})
    segmentations = outputs[0][0]

    confidence_threshold = 0.01
    detected_objects = []
    detected_classes = []

    for segmentation in segmentations:
        confidence = segmentation[4]
        if confidence > confidence_threshold:
            class_probs = segmentation[5:5 + len(class_names)]
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            detected_objects.append({"class": class_label, "confidence": float(confidence)})
            detected_classes.append(class_label)

    return detected_objects, Counter(detected_classes)

cap = cv2.VideoCapture(1)  # 웹캠 번호
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

detected_list = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    frame_count += 1

    if frame_count % 10 == 0:  # 10 프레임 객체 감지 실행. 수정.
        detected_objects, class_counts = detect_objects(frame)
        detected_list = detected_objects  # 리스트 갱신

    for i, obj in enumerate(detected_list):
        label = f"{obj['class']}: {obj['confidence']:.2f}"
        cv2.putText(frame, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Live Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 'q' 누르면 캡처 및 저장
        timestamp = int(time.time())
        filename = f"captured_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"이미지 저장 완료: {filename}")

    elif key == ord('e'):
        detected_list = []
        print("감지된 객체 초기화 완료")

    elif key == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()


# 수정 예정 사항:
# 실시간 캠 켜져 있기만.
# q 버튼 누르면 화면 캡쳐. 저장 기능 빼버리고
# r 버튼 누르면 다시 초기 화면