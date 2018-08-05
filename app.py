from flask import Flask, render_template, request
import json

from Detector import Detector

from PIL import Image
from io import BytesIO

SERVER_IP = '0.0.0.0'
SERVER_PORT = '8000'

PATH_TO_CKPT = '../RCNN_mask/MODEL/mask_rcnn_inception_v2_coco_2018_01_28/' + 'frozen_inference_graph.pb'
PATH_TO_LABELS = '../RCNN_mask/' + 'mscoco_label_map.pbtxt'
NUM_CLASSES = 90

app = Flask(__name__)

def process_results(inference_result):
  boxes = inference_result["detection_boxes"]

  threshold = 0.45
  detected_objects = []
  for i in range(boxes.shape[0]):
    score = float(inference_result["detection_scores"][i])

    if score >= threshold:
      box = tuple(boxes[i].tolist())
      obj = {}
      clazz = int(inference_result["detection_classes"][i])
      
      obj["class"] = clazz
      obj["score"] = score
      obj["box"] = box

      detected_objects.append(obj)

  return detected_objects

@app.route('/')
def index():
  return "Service is running"

@app.route('/infere', methods=["POST"])
def infere():
  if request.method == 'POST':
    if 'img' not in request.files:
      print('image was not submited')
      return request.url
    img_file = request.files['file']
    if img_file.filename == '':
      print('image was not submited')
      return request.url
    img = Image.open(BytesIO(img_file.read()))
    img = frame.convert('RGB')
    img = np.asarray(frame, dtype="uint8")

    detector_output = detector.infere(img)

    resulting_dict = process_results(detector_output)
    return json.dumps(resulting_dict)
  return request.url

if __name__ == '__main__':
  detector = Detector(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES)
  app.run(port=SERVER_PORT, host=SERVER_IP)
