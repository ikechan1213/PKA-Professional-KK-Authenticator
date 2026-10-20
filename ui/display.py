
import cv2


def draw_centering_result(frame, horizontal, vertical):
    """
    カメラ映像にセンタリング結果を描画する。
    """
    cv2.putText(
        frame,
        f"Horizontal: {horizontal}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Vertical: {vertical}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    return frame


def draw_card_outline(frame, corners):
    """
    カメラ映像にカードの輪郭を描画する。

    corners:
        カードの四隅の座標。
        例: [(100, 80), (400, 90), (390, 500), (90, 490)]
    """
    if corners is None or len(corners) != 4:
        return frame

    points = [
        (int(point[0]), int(point[1]))
        for point in corners
    ]

    # 四隅を順番につないでカードの枠を描画する
    for i in range(4):
        start = points[i]
        end = points[(i + 1) % 4]

        cv2.line(
            frame,
            start,
            end,
            (0, 255, 0),
            2
        )

    return frame
