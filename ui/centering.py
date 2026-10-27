
import cv2
import numpy as np


def calculate_centering(outer_corners, inner_corners):
    """
    外枠と内枠の四隅からセンタリング比率を計算する。

    corners の順番:
        左上、右上、右下、左下

    戻り値:
        左右・上下の余白、センタリング比率
    """

    if outer_corners is None or inner_corners is None:
        return None

    if len(outer_corners) != 4 or len(inner_corners) != 4:
        raise ValueError("外枠と内枠にはそれぞれ4点が必要です")

    outer = np.asarray(outer_corners, dtype=np.float32)
    inner = np.asarray(inner_corners, dtype=np.float32)

    # 外枠を正面から見た長方形に変換する
    width = 1000
    height = 1400

    destination = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(outer, destination)

    # 内枠の四隅も同じ座標系に変換する
    inner_rectified = cv2.perspectiveTransform(
        inner.reshape(1, 4, 2),
        matrix
    )[0]

    # 内枠の四隅
    top_left = inner_rectified[0]
    top_right = inner_rectified[1]
    bottom_right = inner_rectified[2]
    bottom_left = inner_rectified[3]

    # 外枠と内枠の間の余白を測定する
    left = (top_left[0] + bottom_left[0]) / 2
    right = (
        (width - top_right[0])
        + (width - bottom_right[0])
    ) / 2

    top = (top_left[1] + top_right[1]) / 2
    bottom = (
        (height - bottom_left[1])
        + (height - bottom_right[1])
    ) / 2

    # 左右・上下それぞれの比率を計算する
    horizontal_total = left + right
    vertical_total = top + bottom

    if horizontal_total <= 0 or vertical_total <= 0:
        raise ValueError("余白を正しく計算できません")

    horizontal_ratio = (
        left / horizontal_total * 100,
        right / horizontal_total * 100
    )

    vertical_ratio = (
        top / vertical_total * 100,
        bottom / vertical_total * 100
    )

    return {
        "left": float(left),
        "right": float(right),
        "top": float(top),
        "bottom": float(bottom),
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio,
    }

