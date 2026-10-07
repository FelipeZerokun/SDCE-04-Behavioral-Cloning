from pathlib import Path

import cv2

from behavioral_cloning.dataset import DrivingSample, load_session


def render_sample(sample: DrivingSample, index: int, total: int):
    images = []
    for image_path in (sample.left, sample.center, sample.right):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        images.append(image)

    combined_image = cv2.hconcat(images)
    display = cv2.copyMakeBorder(
        combined_image,
        top=90,
        bottom=0,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )

    frame_text = f"Frame: {index + 1}/{total}"
    telemetry_text = (
        f"Steering: {sample.steering:+.3f}   "
        f"Throttle: {sample.throttle:.3f}   "
        f"Brake: {sample.brake:.3f}   "
        f"Speed: {sample.speed:.3f}"
    )

    cv2.putText(
        display,
        frame_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        telemetry_text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return display


if __name__ == "__main__":
    csv_path = Path("data/raw/normal_lap_01/driving_log.csv")
    samples = load_session(csv_path)
    index = 0
    window_name = "Left | Center | Right"

    needs_redraw = True

    try:
        while True:
            if needs_redraw:
                display = render_sample(samples[index], index, len(samples))
                cv2.imshow(window_name, display)
                needs_redraw = False

            key = cv2.waitKey(30)

            try:
                window_open = cv2.getWindowProperty(
                    window_name, cv2.WND_PROP_VISIBLE
                ) >= 1
            except cv2.error:
                # Some OpenCV backends raise if the window was destroyed.
                window_open = False

            if not window_open:
                break

            if key == -1:
                continue

            key = key & 0xFF

            if key == ord("q") or key == 27:
                break

            previous_index = index

            if key == ord("d"):
                index = min(index + 1, len(samples) - 1)
            elif key == ord("a"):
                index = max(index - 1, 0)

            needs_redraw = index != previous_index

    finally:
        cv2.destroyAllWindows()