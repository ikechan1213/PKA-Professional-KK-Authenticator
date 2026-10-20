
import cv2

from camera.camera import open_camera, read_frame, release_camera
from ui.display import draw_centering_result, draw_card_outline


def main():
    cap = open_camera()

    try:
        while True:
            frame = read_frame(cap)

            # 現在はダミー値
            frame = draw_centering_result(
                frame,
                "52/48",
                "55/45"
            )

            # カードの輪郭を描画（現在は仮の座標）
            
            # TODO: お兄さん側のカード検出機能と接続する
            corners = None

            # カードが検出された場合だけ輪郭を描画する
            if corners is not None:
                frame = draw_card_outline(frame, corners)

            # 映像を表示
            cv2.imshow("Pokemon Card Centering Checker", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        release_camera(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
