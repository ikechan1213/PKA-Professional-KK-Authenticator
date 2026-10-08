import cv2

from camera.camera import open_camera, read_frame, release_camera
from ui.display import draw_centering_result


def main():
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        # 現在はダミー値
        frame = draw_centering_result(
            frame,
            "52/48",
            "55/45"
        )

        cv2.imshow("Pokemon Card Centering Checker", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    release_camera(cap)


if __name__ == "__main__":
    main()