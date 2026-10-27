
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


def debug_inner_frame(image):
    """
    透視変換後のカード画像を分析し、
    内側の境界を検出するための調査画像を保存する。

    この段階では境界を確定せず、
    色の違いや輪郭を目で確認する。
    """

    # 1. 元画像をコピーする
    # 元の画像を変更しないようにするため
    original = image.copy()

    # 2. BGR形式からHSV形式に変換する
    # HSVでは色相・彩度・明るさを分けて調べられる
    hsv = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2HSV
    )

    # 3. 彩度の画像を取り出す
    # 彩度が高い部分ほど、色が鮮やかな傾向がある
    saturation = hsv[:, :, 1]

    # 彩度の違いを白黒画像として表示する
    saturation_view = cv2.cvtColor(
        saturation,
        cv2.COLOR_GRAY2BGR
    )

    # 4. 彩度にしきい値を設定して白黒に分ける
    # まずは調査用の仮の値として60を使う
    _, saturation_mask = cv2.threshold(
        saturation,
        60,
        255,
        cv2.THRESH_BINARY
    )

    # 5. カード内の模様や文字などの輪郭を検出する
    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    # 6. 輪郭を元画像に重ねる
    edge_overlay = original.copy()

    edge_overlay[edges > 0] = (0, 255, 0)

    # 7. 各画像を保存する
    cv2.imwrite(
        "debug_inner_original.jpg",
        original
    )

    cv2.imwrite(
        "debug_inner_saturation.jpg",
        saturation_view
    )

    cv2.imwrite(
        "debug_inner_mask.jpg",
        saturation_mask
    )

    cv2.imwrite(
        "debug_inner_edges.jpg",
        edges
    )

    cv2.imwrite(
        "debug_inner_overlay.jpg",
        edge_overlay
    )

    # 8. 画像サイズを表示する
    height, width = original.shape[:2]

    print("内側境界の調査を開始しました")
    print("画像サイズ:", width, "x", height)
    print("調査用画像を5枚保存しました")

    print("debug_inner_original.jpg")
    print("debug_inner_saturation.jpg")
    print("debug_inner_mask.jpg")
    print("debug_inner_edges.jpg")
    print("debug_inner_overlay.jpg")



def detect_inner_boundary_candidates(image):
    """
    カードの内側の境界候補を抽出する。

    注意：
    この関数は境界候補を探すための試作版。
    検出結果を正しい印刷領域と確定するものではない。
    """

    # 1. 元画像をコピーする
    original = image.copy()

    # 2. BGR形式からHSV形式に変換する
    hsv = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2HSV
    )

    # 3. 彩度のチャンネルを取り出す
    saturation = hsv[:, :, 1]

    # 4. 彩度60を基準に白黒画像を作る
    # 白：彩度が60を超える場所
    # 黒：それ以外の場所
    _, mask = cv2.threshold(
        saturation,
        60,
        255,
        cv2.THRESH_BINARY
    )

    # 5. 小さな隙間をつなげるためのカーネルを作る
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (11, 11)
    )

    # 6. モルフォロジー処理で細かい隙間を埋める
    closed_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # 7. 白い領域の外側の輪郭を検出する
    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 8. 輪郭が見つからなかった場合は終了する
    if not contours:
        print("内側境界の候補が見つかりませんでした")

        cv2.imwrite(
            "debug_inner_candidate_mask.jpg",
            closed_mask
        )

        return None

    # 9. 面積が最大の輪郭を選ぶ
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # 10. 輪郭の面積と画像全体の面積を調べる
    contour_area = cv2.contourArea(
        largest_contour
    )

    image_height, image_width = original.shape[:2]
    image_area = image_width * image_height

    area_ratio = contour_area / image_area

    print("最大輪郭の面積:", contour_area)
    print("画像全体に対する割合:", round(area_ratio, 4))

    # 11. 最大輪郭を囲む長方形を求める
    x, y, width, height = cv2.boundingRect(
        largest_contour
    )

    print(
        "境界候補の外接矩形:",
        f"x={x}, y={y}, width={width}, height={height}"
    )

    # 12. 確認用画像に最大輪郭を緑色で描画する
    preview = original.copy()

    cv2.drawContours(
        preview,
        [largest_contour],
        -1,
        (0, 255, 0),
        2
    )

    # 13. 最大輪郭を囲む長方形を青色で描画する
    cv2.rectangle(
        preview,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )

    # 14. 調査用画像を保存する
    cv2.imwrite(
        "debug_inner_candidate_mask.jpg",
        closed_mask
    )

    cv2.imwrite(
        "debug_inner_candidate.jpg",
        preview
    )

    # 15. 結果を表示する
    print("境界候補の調査画像を保存しました")
    print("debug_inner_candidate_mask.jpg")
    print("debug_inner_candidate.jpg")

    # 次の段階で使えるように輪郭を返す
    return largest_contour


def detect_inner_quad(image, contour):
    """
    検出した内側境界の輪郭から、
    四角形の候補を推定して確認用画像を保存する。

    この段階では、評価に使う境界としてはまだ確定しない。
    """

    if contour is None or len(contour) < 4:
        print("四角形の推定に必要な輪郭がありません")
        return None

    # 1. 輪郭の周囲の長さを計算する
    perimeter = cv2.arcLength(
        contour,
        True
    )

    # 2. 輪郭を少ない点で近似する
    # epsilonが小さいほど元の輪郭に近くなる
    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    print("輪郭近似後の頂点数:", len(approx))

    # 3. 近似した輪郭が4点なら、その点を候補にする
    if len(approx) == 4 and cv2.isContourConvex(approx):
        corners = approx.reshape(4, 2).astype(np.float32)
        method = "輪郭近似"

    else:
        # 4. 4点にならなければ、回転した長方形で近似する
        # これは代替候補であり、正しい境界とは限らない
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(np.float32)
        method = "最小外接回転矩形"

    # 5. 四隅を時計回りに並べる
    corners = order_corners(corners)

    # 6. 元画像をコピーして描画する
    preview = image.copy()

    polygon = np.round(
        corners
    ).astype(np.int32).reshape((-1, 1, 2))

    # 四角形の候補を黄色で描画する
    cv2.polylines(
        preview,
        [polygon],
        True,
        (0, 255, 255),
        3
    )

    # 各頂点に赤い丸を描画する
    for index, point in enumerate(corners):
        x = int(round(point[0]))
        y = int(round(point[1]))

        cv2.circle(
            preview,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            preview,
            str(index + 1),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # 7. 確認用画像を保存する
    cv2.imwrite(
        "debug_inner_quad.jpg",
        preview
    )

    # 8. 四隅の座標と使用した方法を表示する
    print("四角形候補の推定方法:", method)
    print("四角形候補の四隅:")

    for index, point in enumerate(corners):
        print(
            f"  頂点{index + 1}: "
            f"x={point[0]:.1f}, y={point[1]:.1f}"
        )

    print("debug_inner_quad.jpg を保存しました")

    return corners


def calculate_centering(image, inner_corners):
    """
    透視変換後のカード画像と内側境界の四隅から、
    上下左右の余白と、それぞれの比率を計算する。

    ここで求めるのは画像上の余白の比率であり、
    PSAの正式な評価結果ではない。
    """

    # 四隅が取得できているか確認する
    if inner_corners is None or len(inner_corners) != 4:
        print("余白を計算できる四隅がありません")
        return None

    # 画像の高さと幅を取得する
    height, width = image.shape[:2]

    # 四隅を左上、右上、右下、左下の順に並べる
    corners = order_corners(inner_corners)

    # 四隅の座標を取り出す
    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    # 左右の境界は、各辺の2点の平均位置で求める
    left_x = (top_left[0] + bottom_left[0]) / 2
    right_x = (top_right[0] + bottom_right[0]) / 2

    # 上下の境界も、各辺の2点の平均位置で求める
    top_y = (top_left[1] + top_right[1]) / 2
    bottom_y = (bottom_left[1] + bottom_right[1]) / 2

    # 画像の端から内側境界までの距離を求める
    left_margin = left_x
    right_margin = (width - 1) - right_x
    top_margin = top_y
    bottom_margin = (height - 1) - bottom_y

    # 計算結果が不正でないか確認する
    margins = [
        left_margin,
        right_margin,
        top_margin,
        bottom_margin
    ]

    if any(margin < 0 for margin in margins):
        print("内側境界が画像の外側にあるため、計算を中止します")
        return None

    # 左右・上下それぞれの合計余白を求める
    horizontal_total = left_margin + right_margin
    vertical_total = top_margin + bottom_margin

    # 余白の合計が0の場合は比率を計算できない
    if horizontal_total <= 0 or vertical_total <= 0:
        print("余白の合計が0のため、比率を計算できません")
        return None

    # 左右それぞれの比率を百分率で計算する
    left_percent = left_margin / horizontal_total * 100
    right_percent = right_margin / horizontal_total * 100

    # 上下それぞれの比率を百分率で計算する
    top_percent = top_margin / vertical_total * 100
    bottom_percent = bottom_margin / vertical_total * 100

    # 結果を表示する
    print("\n===== センタリング測定結果（試作版） =====")

    print(f"左の余白:   {left_margin:.2f} px")
    print(f"右の余白:   {right_margin:.2f} px")
    print(f"上の余白:   {top_margin:.2f} px")
    print(f"下の余白:   {bottom_margin:.2f} px")

    print("\n----- 左右の比率 -----")
    print(f"左: {left_percent:.2f}%")
    print(f"右: {right_percent:.2f}%")
    print(f"比率: {left_percent:.2f}:{right_percent:.2f}")

    print("\n----- 上下の比率 -----")
    print(f"上: {top_percent:.2f}%")
    print(f"下: {bottom_percent:.2f}%")
    print(f"比率: {top_percent:.2f}:{bottom_percent:.2f}")

    return {
        "left_margin": left_margin,
        "right_margin": right_margin,
        "top_margin": top_margin,
        "bottom_margin": bottom_margin,
        "left_percent": left_percent,
        "right_percent": right_percent,
        "top_percent": top_percent,
        "bottom_percent": bottom_percent
    }



def draw_centering_overlay(image, inner_corners, result):
    """
    内側境界と上下左右のセンタリング比率を画像に描画する。
    画像ファイルを保存し、表示用の画像を返す。
    """

    if inner_corners is None or result is None:
        print("描画に必要な測定結果がありません")
        return None

    # 元画像を変更しないようにコピーする
    preview = image.copy()

    height, width = preview.shape[:2]

    # 四隅を左上、右上、右下、左下の順に並べる
    corners = order_corners(inner_corners)

    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]

    # 内側境界を緑色で描画する
    polygon = np.round(corners).astype(np.int32).reshape((-1, 1, 2))

    cv2.polylines(
        preview,
        [polygon],
        True,
        (0, 255, 0),
        2
    )

    # 各辺の中央付近を求める
    top_mid = (
        int(round((top_left[0] + top_right[0]) / 2)),
        int(round((top_left[1] + top_right[1]) / 2))
    )

    bottom_mid = (
        int(round((bottom_left[0] + bottom_right[0]) / 2)),
        int(round((bottom_left[1] + bottom_right[1]) / 2))
    )

    left_mid = (
        int(round((top_left[0] + bottom_left[0]) / 2)),
        int(round((top_left[1] + bottom_left[1]) / 2))
    )

    right_mid = (
        int(round((top_right[0] + bottom_right[0]) / 2)),
        int(round((top_right[1] + bottom_right[1]) / 2))
    )

    # 上下の余白を示す線（青色）
    cv2.line(
        preview,
        (top_mid[0], 0),
        top_mid,
        (255, 0, 0),
        2
    )

    cv2.line(
        preview,
        bottom_mid,
        (bottom_mid[0], height - 1),
        (255, 0, 0),
        2
    )

    # 左右の余白を示す線（オレンジ色）
    cv2.line(
        preview,
        (0, left_mid[1]),
        left_mid,
        (0, 165, 255),
        2
    )

    cv2.line(
        preview,
        right_mid,
        (width - 1, right_mid[1]),
        (0, 165, 255),
        2
    )

    
    # 表示設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_color = (0, 165, 255)

    # 上辺の灰色部分に表示する文字列
    left_text = f"Left: {result['left_percent']:.2f}%"
    top_text = f"Top: {result['top_percent']:.2f}%"
    right_text = f"Right: {result['right_percent']:.2f}%"

    # 文字のサイズを取得する
    left_size = cv2.getTextSize(
        left_text, font, scale, thickness
    )[0]

    top_size = cv2.getTextSize(
        top_text, font, scale, thickness
    )[0]

    right_size = cv2.getTextSize(
        right_text, font, scale, thickness
    )[0]

    # 左端・中央・右端の表示位置を計算する
    left_x = 8
    top_x = max(8, (width - top_size[0]) // 2 + 50)
    right_x = max(8, width - right_size[0] - 8)

    # 上辺の灰色部分に3つの比率を描画する
    # 文字が上端から切れないように、y座標に余裕を持たせる
    text_y = 20

    cv2.putText(
        preview,
        left_text,
        (left_x, text_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )
#白い縁取りを先に描画する
    cv2.putText(
    preview,
    top_text,
    (top_x, text_y),
    font,
    scale,
    (255, 255, 255),
    thickness + 3,
    cv2.LINE_AA
)

# 青い文字を上から描画する
    cv2.putText(
        preview,
        top_text,
        (top_x, text_y),
        font,
        scale,
        (255, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    cv2.putText(
        preview,
        right_text,
        (right_x, text_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    # 下の比率を画像下部に表示する
    
    # Bottomの文字列を作成する
    bottom_text = f"Bottom: {result['bottom_percent']:.2f}%"

    # 文字列の幅と高さを取得する
    bottom_size = cv2.getTextSize(
        bottom_text,
        font,
        scale,
        thickness
    )[0]

    # 下辺の右端に表示する位置を計算する
    bottom_x = width - bottom_size[0] - 80
    bottom_y = height - 7

    # Bottomの比率を描画する
    # 白い縁取りを先に描画する
    cv2.putText(
        preview,
        bottom_text,
        (bottom_x, bottom_y),
        font,
        scale,
        (255, 255, 255),
        thickness + 3,
        cv2.LINE_AA
    )

    # 青い文字を上から描画する
    cv2.putText(
        preview,
        bottom_text,
        (bottom_x, bottom_y),
        font,
        scale,
        (255, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    # 測定結果の画像を保存する
    output_path = "debug_centering_overlay.jpg"

    saved = cv2.imwrite(
        output_path,
        preview
    )

    if saved:
        print(f"{output_path} を保存しました")
    else:
        print(f"{output_path} の保存に失敗しました")

    # PC画面に測定結果を新しいウィンドウで表示する
    
    
    # 測定結果の画像を保存する
    # 保存される画像は縮小しない
    output_path = "debug_centering_overlay.jpg"

    saved = cv2.imwrite(
        output_path,
        preview
    )

    if saved:
        print(f"{output_path} を保存しました")
    else:
        print(f"{output_path} の保存に失敗しました")

    # 画面表示用の画像だけを縮小する
    display_max_width = 600
    display_max_height = 700

    height, width = preview.shape[:2]

    # ウィンドウに収まる縮小率を計算する
    scale = min(
        display_max_width / width,
        display_max_height / height,
        1.0
    )

    display_image = cv2.resize(
        preview,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    # 縮小した画像を新しいウィンドウに表示する
    cv2.imshow(
        "Centering Measurement",
        display_image
    )

    return preview
    
    

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
    # 透視変換後のカードを分析する
    debug_inner_frame(warped_card)
    # 内側の境界候補を検出する
    inner_contour = detect_inner_boundary_candidates(
    warped_card
    )

    # 検出した輪郭から四角形の候補を推定する
    if inner_contour is not None:
        inner_corners = detect_inner_quad(
        warped_card,
        inner_contour
    )
    else:
        inner_corners = None
        print("内側の四角形候補を作成できませんでした")

    
# 内側境界の四隅が取得できた場合だけ余白を計算する
    if inner_corners is not None:
        centering_result = calculate_centering(
            warped_card,
            inner_corners
        )
    else:
        centering_result = None
        print("四隅が取得できないため、余白の計算をスキップします")

        # 余白と内側境界を画像に描画する
    if centering_result is not None:
        draw_centering_overlay(
        warped_card,
        inner_corners,
        centering_result
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