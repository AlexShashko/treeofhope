import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import flask as fl
import base64

import keras
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

from retinanet.metrics import smooth_l1, focal
from retinanet.model import RetinaNet, add_bboxes_to_model
from utils import draw_box, draw_caption, label_color

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = fl.Flask(__name__)
prediction_model = None


def load_model(number_of_classes: int, model_path: str):
    global prediction_model

    resina = RetinaNet(number_of_classes)
    resina.model.compile(
        loss={
            "regression": smooth_l1(),
            "classification": focal()
        },
		optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )
    prediction_model = add_bboxes_to_model(resina.model)

    prediction_model.load_weights(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if fl.request.method == "POST":
        req_image = fl.request.get_json()["img"]
        name = fl.request.get_json()["name"]

        with open("temp.jpg", "wb") as f:
            f.write(base64.decodebytes(req_image.encode()))

        image = imread("temp.jpg")

        # size = image.shape
        ik = np.zeros((1, 800, 800, 3), dtype=np.uint8)

        image = resize(image, (800, 800), mode="constant", preserve_range=True)
        ik[0] = image
        image = ik[0]

        labels_to_names = {2: 'maple',
                           1: 'juniper',
                           0: 'oak',
                           3: 'spruce',
                           4: 'thuja',
                           5: 'birch',
                           6: 'mistletoe'}

        boxes, scores, labels = prediction_model.predict(np.expand_dims(image, axis=0))

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

	        if abs(box[0] - box[2]) < image.shape[0] * 0.1 or abs(box[1] - box[3]) < image.shape[0] * 0.1:
	          continue
	        if abs(box[0] - box[2]) > image.shape[0] * 0.8 or abs(box[1] - box[3]) > image.shape[0] * 0.8:
	          continue
	        if abs(box[0] - box[2]) > abs(box[1] - box[3]):
	          continue

            color = label_color(label)

            b = box.astype(int)
            draw_box(image, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(image, b, caption)

        imsave("temp.jpg", image)

        with open(f"data/{name}.png", "rb") as f_img:
            encoded = base64.b64encode(f_img.read())

        data["success"] = True
        data["image"] = encoded.decode()

    return fl.jsonify(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-NC', '--number_classes', required=True, help="Enter number of classes in RetinaNet model.")
    parser.add_argument('-MP', '--model_path', default="weights/weights.h5", help="Enter path to model weights.")
    parser.add_argument('-hs', '--host', default="0.0.0.0", help="Enter API host.")
    parser.add_argument('-p', '--port', default="5000", help="Enter API port.")
    parser.add_argument('-d', "--debug", default=False, help="API debug option.")
    args = parser.parse_args()

    load_model(int(args.number_classes), args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
