
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

    print("edges.shape:", edges.shape)
    print("points.shape:", points.shape)
    print("p1. shape:", p1.shape)
    print("p2. shape:", p2.shape)
    print("points:", points[:5])

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


def warp_card(image, corners):
    """
    カードの四隅を使って透視変換し、
    正面から見た長方形の画像を作成する。
    """

    # 四隅を時計回りに並べる
    corners = order_corners(corners).astype(np.float32)

    # 各辺の長さを計算する
    top_width = np.linalg.norm(corners[1] - corners[0])
    bottom_width = np.linalg.norm(corners[2] - corners[3])
    left_height = np.linalg.norm(corners[3] - corners[0])
    right_height = np.linalg.norm(corners[2] - corners[1])

    # 上下・左右で長い方を出力サイズにする
    output_width = int(round(max(top_width, bottom_width)))
    output_height = int(round(max(left_height, right_height)))

    if output_width < 2 or output_height < 2:
        raise ValueError("透視変換後の画像サイズが不正です")

    # 変換後の四隅を定義する
    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype=np.float32)

    # 変換前と変換後の四隅から変換行列を求める
    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    # 透視変換を実行する
    warped = cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC
    )

    return warped


def detect_outer_boundary(gray, edges, corners, band=35):
    """
    カード外周の外側にあるエッジから境界候補を検出する。
    下辺は区間ごとに最も近いエッジを選んで推定する。
    """

    corners = order_corners(corners)

    ys, xs = np.where(edges > 0)
    points = np.column_stack((xs, ys)).astype(np.float32)

    lines = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]

        direction = p2 - p1
        length = np.linalg.norm(direction)

        if length < 1:
            lines.append(None)
            continue

        unit = direction / length
        relative = points - p1

        # 辺に沿った位置
        projection = relative @ unit

        # 辺からの符号付き距離
        signed_distance = (
            direction[0] * relative[:, 1]
            - direction[1] * relative[:, 0]
        ) / length

        # 四角形の外側にある点だけを抽出
        distance = -signed_distance

        
        # 下辺は、明るさの変化を使って境界を探す
        is_bottom_edge = (i == 2)

        if is_bottom_edge:
            # 下辺は現在の線より外側25ピクセルまで探索
            search_band = 60
        else:
            search_band = band

        distance = -signed_distance

        mask = (
            (projection > length * 0.10)
            & (projection < length * 0.90)
            & (distance > 2)
            & (distance < search_band)
        )

        selected = points[mask]
        selected_projection = projection[mask]
        selected_distance = distance[mask]

        if is_bottom_edge:
            # グレースケール画像から明るさの変化を計算
            gradient_y = cv2.Sobel(
                gray,
                cv2.CV_32F,
                0,
                1,
                ksize=3
            )

            # 下辺を横方向に30区間に分割
            number_of_bins = 30
            bin_width = length * 0.8 / number_of_bins
            chosen_points = []

            for bin_index in range(number_of_bins):
                bin_center_ratio = (
                    (bin_index + 0.5) / number_of_bins
                )
                # 中央35〜65%は文字などの影響を避けるため除外
                if 0.35 < bin_center_ratio < 0.65:
                    continue
                start = length * 0.1 + bin_index * bin_width
                end = start + bin_width

                in_bin = (
                    (selected_projection >= start)
                    & (selected_projection < end)
                )

                if not np.any(in_bin):
                    continue

                bin_points = selected[in_bin]
                bin_distances = selected_distance[in_bin]

                # 元の線から外側へ進みながら、
                # 明るさの変化が強いエッジを優先する
                best_point = None
                best_score = -1

                for point, distance_value in zip(
                    bin_points, bin_distances
                ):
                    x = int(round(point[0]))
                    y = int(round(point[1]))

                    if (
                        x < 1 or x >= gray.shape[1] - 1
                        or y < 1 or y >= gray.shape[0] - 1
                    ):
                        continue

                    # 下辺付近で縦方向の明るさ変化を評価
                    score = abs(float(gradient_y[y, x]))

                    # 遠い点を少しだけ不利にする
                    score /= (1.0 + distance_value * 0.01)

                    if score > best_score:
                        best_score = score
                        best_point = point

                if best_point is not None:
                    chosen_points.append(best_point)
                    bottom_debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    for point in chosen_points:
                        x = int(round(point[0]))
                        y = int(round(point[1]))
                        cv2.circle(
                            bottom_debug,
                            (x, y),
                            4,
                            (0, 0, 255),
                            -1
                        )
                    cv2.imwrite(
                        "debug_bottom_points.jpg",
                        bottom_debug
                    )
            selected = np.array(
                chosen_points,
                dtype=np.float32
            )

            if len(selected) < 8:
                print("下辺: 境界点が不足しています")
                lines.append(None)
                continue

        elif len(selected) < 20:
            print(f"辺{i + 1}: 外側のエッジ点が不足")
            lines.append(None)
            continue
        
        
        # 選択した点から直線を推定
        if len(selected) < 2:
            print(f"辺{i + 1}: 直線推定に使える点が不足")
            lines.append(None)
            continue

        vx, vy, x0, y0 = cv2.fitLine(
            selected,
            cv2.DIST_L1,
            0,
            0.01,
            0.01
        ).flatten()

        line = np.array(
            [vx, vy, x0, y0],
            dtype=np.float64
        )

        lines.append(line)

        print(
            f"辺{i + 1}: 直線推定に使用した点数="
            f"{len(selected)}"
        )

        print(
            f"辺{i + 1}: 直線推定に使用した点数="
            f"{len(selected)}"
        )

    # 隣接する直線の交点から四隅を求める
    outer_corners = []

    for i in range(4):
        previous_line = lines[(i - 1) % 4]
        current_line = lines[i]

        corner = intersect_lines(previous_line, current_line)

        if corner is None:
            print("外側の境界は4つの角を確定できませんでした")
            return None

        outer_corners.append(corner)

    result = np.array(outer_corners, dtype=np.float32)

    # 画像外にある候補は採用しない
    height, width = edges.shape

    if (
        np.any(result[:, 0] < 0)
        or np.any(result[:, 0] >= width)
        or np.any(result[:, 1] < 0)
        or np.any(result[:, 1] >= height)
    ):
        print("外側の境界候補が画像の範囲外です")
        return None

    return result


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

    
    # 背景とカード本体の境界候補を独立して検出
    outer_corners = detect_outer_boundary(
    gray,
    boundary_edges,
    refined_corners,
    band=35
)

    if outer_corners is not None:
        outer_poly = np.round(
            outer_corners
        ).astype(np.int32).reshape((-1, 1, 2))

        # オレンジ色で外側の境界候補を表示
        cv2.polylines(
            debug,
            [outer_poly],
            True,
            (0, 165, 255),
            3
        )

        print("外側の境界候補:", outer_corners.tolist())
    else:
        print("外側の境界は検出できませんでした")
    
    # 透視変換に使う四隅を選択する
    if outer_corners is not None:
        warp_corners = outer_corners
        print("透視変換: 外側の境界を使用します")
    else:
        warp_corners = refined_corners
        print("透視変換: 補正後の四隅を使用します")

    # カードを正面向きの長方形に変換する
    warped_card = warp_card(small, warp_corners)

    # 変換後の画像を保存する
    cv2.imwrite(
        "debug_warped_card.jpg",
        warped_card
    )

    print(
        "透視変換後の画像サイズ:",
        warped_card.shape[1],
        "x",
        warped_card.shape[0]
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

    
    # 検出結果の確認画面
    cv2.imshow("Card Edge Refinement", preview)

    # 透視変換したカードの確認画面
    warped_preview = cv2.resize(
        warped_card,
        None,
        fx=0.6,
        fy=0.6,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow(
        "Perspective Transform",
        warped_preview
    )

    # どちらかの画面でキーを押すと終了
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 最終結果に使う四隅を決める
    # 外側の境界を検出できた場合は、そちらを採用する
    if outer_corners is not None:
        final_corners = outer_corners
    else:
        # 外側の境界を検出できなかった場合は従来の四隅を使う
        final_corners = refined_corners
    # 元画像の座標に戻す
    result = [
        (
            int(round(point[0] / scale)),
            int(round(point[1] / scale))
        )
        for point in final_corners
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