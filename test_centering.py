
from ui.centering import calculate_centering


# 仮の外枠と内枠の座標
# 順番は左上、右上、右下、左下
outer_corners = [
    (50, 50),
    (550, 50),
    (550, 750),
    (50, 750),
]

inner_corners = [
    (80, 85),
    (515, 90),
    (510, 715),
    (85, 710),
]

result = calculate_centering(
    outer_corners,
    inner_corners
)

print("左右の余白:", result["left"], result["right"])
print("上下の余白:", result["top"], result["bottom"])

left_pct, right_pct = result["horizontal_ratio"]
top_pct, bottom_pct = result["vertical_ratio"]

print(f"左右センタリング: {left_pct:.1f}/{right_pct:.1f}")
print(f"上下センタリング: {top_pct:.1f}/{bottom_pct:.1f}")

