import argparse
import base64
import os
from datetime import UTC, datetime
from io import BytesIO

import h5py
import numpy as np
import socketio
from flask import Flask
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from keras import __version__ as keras_version
from keras.models import load_model
from PIL import Image

from behavioral_cloning.preprocessing import preprocess_rgb

sio = socketio.Server(async_mode="gevent") #Replaced -> sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString))).convert("RGB")
        image_array = preprocess_rgb(np.asarray(image, dtype=np.uint8))

        prediction = model.predict(image_array[None, ...], verbose=0)
        steering_angle = float(prediction[0, 0])

        throttle = float(np.clip(controller.update(float(speed)), -1.0, 1.0))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.now(UTC).strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    with h5py.File(args.model, mode="r") as model_file:
        model_version = model_file.attrs.get("keras_version")

    if isinstance(model_version, bytes):
        model_version = model_version.decode("utf-8")

    if model_version is None:
        print("The model does not contain Keras version metadata.")
    elif model_version != keras_version:
        print(
            f"Installed Keras version: {keras_version}; "
            f"model saved with: {model_version}"
        )

    model = load_model(args.model, compile=False)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        os.makedirs(args.image_folder, exist_ok=False)

        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    app = socketio.WSGIApp(sio, app)

    server = pywsgi.WSGIServer(
        ("127.0.0.1", 4567),
        app,
        handler_class=WebSocketHandler,
    )

    print("Listening for the simulator on 127.0.0.1:4567", flush=True)
    server.serve_forever()
