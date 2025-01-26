from flask import Flask, request, render_template, jsonify
import onnxruntime as ort
import numpy as np
import cv2
from collections import Counter
import requests


app = Flask(__name__)

model_path = "chaesiktak_weights.onnx"
class_names_path = "chaesiktak_classes.txt"


def load_class_names(class_file):
    with open(class_file, "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names

class_names = load_class_names(class_names_path)


def preprocess_image_from_url(image_url, input_size=640):
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # 배치 추가. 1장씩 입력 가정
    return image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/segment', methods=['POST'])
def segment():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400
    
    image_url = data['image_url']
    try:
        input_tensor = preprocess_image_from_url(image_url)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    outputs = session.run([output_name], {input_name: input_tensor})
    segmentations = outputs[0][0]
    confidence_threshold = 0.01

    segmentation_objects = []
    segmentation_classes = []
    for segmentation in segmentations:
        confidence = segmentation[4]
        if confidence > confidence_threshold:
            class_probs = segmentation[5:5 + len(class_names)]
            class_id = np.argmax(class_probs)
            class_label = class_names[class_id]
            segmentation_objects.append({"class": class_label, "confidence": float(confidence)})
            segmentation_classes.append(class_label)

    class_counts = Counter(segmentation_classes)
    count_summary = {cls: count for cls, count in class_counts.items()}

    return jsonify({"segmentations": segmentation_objects, "counts": count_summary})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)