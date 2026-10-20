
import cv2
import numpy as np


def order_corners(points):
    """4点を時計回りに並べる。"""

    points = np.asarray(points, dtype=np.float32)
    center = points.mean(axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0]
    )

    return points[np.argsort(angles)]



def fit_edge_line(edges, p1, p2, band=18):
    """
    四角形の各辺の近くから直線を推定する。
    まず四角形の外側にあるエッジを優先する。
    """

    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1:
        return None

    unit = direction / length

    ys, xs = np.where(edges > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)

    relative = points - p1

    # 辺に沿った位置
    projection = relative @ unit

    # 辺からの符号付き距離
    signed_distance = (
        direction[0] * relative[:, 1]
        - direction[1] * relative[:, 0]
    ) / length

    # 四角形の外側にあるエッジを優先
    outside_mask = (
        (projection > length * 0.15)
        & (projection < length * 0.85)
        & (signed_distance < 0)
        & (signed_distance > -band)
    )

    selected = points[outside_mask]

    # 外側の点が少なければ、両側から探す
    if len(selected) < 20:
        distance = np.abs(signed_distance)

        fallback_mask = (
            (projection > length * 0.15)
            & (projection < length * 0.85)
            & (distance < band)
        )

        selected = points[fallback_mask]

    if len(selected) < 20:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        selected,
        cv2.DIST_L1,
        0,
        0.01,
        0.01
    ).flatten()

    return np.array(
        [vx, vy, x0, y0],
        dtype=np.float64
    )


def intersect_lines(line1, line2):
    """2本の直線の交点を計算する。"""

    if line1 is None or line2 is None:
        return None

    vx1, vy1, x1, y1 = line1
    vx2, vy2, x2, y2 = line2

    matrix = np.array([
        [vx1, -vx2],
        [vy1, -vy2]
    ])

    determinant = np.linalg.det(matrix)

    if abs(determinant) < 1e-6:
        return None

    rhs = np.array([x2 - x1, y2 - y1])
    t, _ = np.linalg.solve(matrix, rhs)

    return np.array(
        [x1 + t * vx1, y1 + t * vy1],
        dtype=np.float32
    )


def detect_card(image):
    """カードの四隅を検出し、辺の直線で補正する。"""

    if image is None:
        raise ValueError("画像が読み込まれていません")

    height, width = image.shape[:2]

    # 処理用画像を縮小
    scale = 1200 / max(height, width)

    small = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 初期候補を検出
    edges = cv2.Canny(blurred, 25, 80)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (9, 9)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = small.shape[0] * small.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        x, y, w, h = cv2.boundingRect(approx)

        if h == 0:
            continue

        ratio = w / h

        if not 0.45 <= ratio <= 0.95:
            continue

        candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        print("カード候補が見つかりませんでした")
        return None

    # 最大面積の候補を初期値として使う
    candidates.sort(key=lambda item: item[0], reverse=True)

    original_corners = order_corners(candidates[0][1])

    # 各辺の近くから直線を推定
    lines = []

    for i in range(4):
        p1 = original_corners[i]
        p2 = original_corners[(i + 1) % 4]

        line = fit_edge_line(
            edges, p1, p2, band=10
        )

        lines.append(line)

    # 隣り合う辺の交点から四隅を補正
    refined_corners = []

    for i in range(4):
        previous_line = lines[(i - 1) % 4]
        current_line = lines[i]

        corner = intersect_lines(
            previous_line, current_line
        )

        if corner is None:
            # 交点が求められない場合は元の角を使う
            corner = original_corners[i]

        refined_corners.append(corner)

    refined_corners = np.array(
        refined_corners, dtype=np.float32
    )

    # 補正前後を比較する画像を作成
    debug = small.copy()
    refined_poly = np.round(
        refined_corners
    ).astype(np.int32).reshape((-1, 1, 2))

    # 補正後：緑色
    cv2.polylines(
        debug, [refined_poly], True,
        (0, 255, 0), 3
    )

    # 補正後の四隅：赤色
    for point in refined_corners:
        x, y = np.round(point).astype(int)

        cv2.circle(
            debug, (x, y), 7,
            (0, 0, 255), -1
        )
        
    # 背景とカードの境界を探すためのエッジ画像
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    boundary_edges = cv2.Canny(blurred, 25, 80)

    # 現在のカード外周をオレンジ色の線で表示する
    # ※これは新たな検出結果ではなく、比較用の表示
    cv2.polylines(
        debug,
        [refined_poly],
        True,
        (0, 165, 255),
        2
    )

    # エッジ画像を保存して、背景との境界を確認できるようにする
    cv2.imwrite("debug_boundary_edges.jpg", boundary_edges)
    cv2.imwrite("debug_card_refined.jpg", debug)

    preview = cv2.resize(
        debug,
        None,
        fx=0.6,
        fy=0.6,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Card Edge Refinement", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 元画像の座標に戻す
    result = [
        (
            int(round(point[0] / scale)),
            int(round(point[1] / scale))
        )
        for point in refined_corners
    ]

    print("補正前の四隅:", original_corners.tolist())
    print("補正後の四隅:", result)

    return result


if __name__ == "__main__":
    image = cv2.imread("images/test_card.jpg")

    if image is None:
        raise SystemExit(
            "画像を読み込めません。パスを確認してください。"
        )

    detect_card(image)