from flask import Flask, request, render_template
import onnxruntime as ort
import numpy as np
import cv2

app = Flask(__name__)

model_path = "chaesiktak_weights.onnx"
class_names_path = "chaesiktak_classes.txt"


def load_class_names(class_file):
    with open(class_file, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

class_names = load_class_names(class_names_path)


def preprocess_image(image_path, input_size=640):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 배치 추가. 1장씩 입력 가정
    return image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image_path = "uploaded_image.jpg"  #
            file.save(image_path)
            input_tensor = preprocess_image(image_path)

            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            outputs = session.run([output_name], {input_name: input_tensor})
            segmentations = outputs[0][0]
            confidence_threshold = 0.01  # 임계값

            segmented_objects = []
            for segmentation in segmentations:
                confidence = segmentation[4]  # 신뢰도 값 5번째 요소
                if confidence > confidence_threshold:
                    class_probs = segmentation[5:5 + len(class_names)]  # 클래스 개수 만큼 신뢰도 값
                    class_id = np.argmax(class_probs)  # 신뢰도 가장 높은 클래스
                    class_label = class_names[class_id]
                    segmented_objects.append(f"Class: {class_label}, Confidence: {confidence:.2f}")

            return render_template('index.html', segmentations = segmented_objects)
    
    return render_template('index.html', segmentations = None)

if __name__ == '__main__':
    app.run(debug=True)