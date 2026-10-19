import argparse
import base64
import hashlib
import os
from datetime import UTC, datetime
from io import BytesIO
from time import perf_counter

import h5py
import eventlet
import eventlet.wsgi
import numpy as np
import socketio
from flask import Flask
from keras import __version__ as keras_version
from keras.models import load_model
from PIL import Image

from behavioral_cloning.preprocessing import preprocess_rgb

sio = socketio.Server(async_mode="eventlet")
app = Flask(__name__)
model = None
diagnostics: dict[str, tuple[str, float, int, int]] = {}
connections: set[str] = set()


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
        callback_start = perf_counter()
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]

        image_bytes = base64.b64decode(imgString)
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        previous_hash, previous_time, previous_repeat, previous_count = diagnostics.get(
            sid, ("", callback_start, 0, 0)
        )
        repeat = previous_repeat + 1 if image_hash == previous_hash else 1
        sequence = previous_count + 1
        interval_ms = (callback_start - previous_time) * 1000
        diagnostics[sid] = (image_hash, callback_start, repeat, sequence)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_array = preprocess_rgb(np.asarray(image, dtype=np.uint8))

        predict_start = perf_counter()
        prediction = model(image_array[None, ...], training=False).numpy()
        predict_ms = (perf_counter() - predict_start) * 1000

        steering_angle = float(prediction[0, 0])

        throttle = float(np.clip(controller.update(float(speed)), -1.0, 1.0))

        send_control(steering_angle, throttle)
        emit_return_ms = (perf_counter() - callback_start) * 1000

        # save frame
        if args.image_folder != '':
            timestamp = datetime.now(UTC).strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        callback_ms = (perf_counter() - callback_start) * 1000
        print(
            f"DIAG t={callback_start:.6f} "
            f"sid={sid} seq={sequence} repeat={repeat} "
            f"interval_ms={interval_ms:.2f} "
            f"image={image_hash} "
            f"speed={float(speed):.3f} "
            f"steering={steering_angle:.6f} "
            f"throttle={throttle:.6f} "
            f"predict_ms={predict_ms:.2f} "
            f"emit_return_ms={emit_return_ms:.2f} "
            f"callback_ms={callback_ms:.2f}",
            flush=True,
        )
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    connections.add(sid)
    diagnostics.pop(sid, None)
    print(f"CONNECT sid={sid} active={len(connections)}", flush=True)
    send_control(0, 0)


@sio.on('disconnect')
def disconnect(sid):
    connections.discard(sid)
    diagnostics.pop(sid, None)
    print(f"DISCONNECT sid={sid} active={len(connections)}", flush=True)


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

    print(
        f"RUN backend={sio.async_mode} model={args.model} target_speed={set_speed} "
        f"Kp={controller.Kp} Ki={controller.Ki} inference=direct "
        f"recording={args.image_folder or 'disabled'}",
        flush=True,
    )
    print("Listening for the simulator on 127.0.0.1:4567", flush=True)
    eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 4567)), app)
