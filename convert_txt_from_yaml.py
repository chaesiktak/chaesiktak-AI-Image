import yaml


yaml_file_path = "data.yaml"
output_txt_path = "chaesiktak_classes.txt"

with open(yaml_file_path, "r") as f:
    data = yaml.safe_load(f)

class_names = data["names"]

with open(output_txt_path, "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("파일 변환 완료.")