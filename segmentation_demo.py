import cv2
import onnxruntime as ort
import numpy as np


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
    detected_classes = []

    for segmentation in segmentations:
        confidence = segmentation[4]
        if confidence > confidence_threshold:
            class_probs = segmentation[5:5 + len(class_names)]
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            if class_label not in detected_classes:  # 중복 방지
                detected_classes.append(class_label)

    return detected_classes

cap = cv2.VideoCapture(1)  # 웹캠 번호
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

detected_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    cv2.namedWindow('Chaesiktak Image analysis', cv2.WINDOW_NORMAL)
    cv2.imshow("Chaesiktak Image analysis", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        detected_objects = detect_objects(frame)
        detected_list = detected_objects
        print("감지된 객체:", detected_list)

        if detected_list:
            annotated_frame = frame.copy()
            for i, obj in enumerate(detected_list):
                cv2.putText(annotated_frame, obj, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Segmentationed", annotated_frame)

    elif key == ord('e'):
        detected_list = []
        print("감지된 객체 초기화 완료")

    elif key == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()