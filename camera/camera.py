import cv2


def open_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("カメラを開けませんでした")

    return cap


def read_frame(cap):
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("カメラ画像を取得できませんでした")

    return frame


def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    release_camera(cap)