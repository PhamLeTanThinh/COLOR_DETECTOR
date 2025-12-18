# color/dominant_color.py
# Copy-đè toàn bộ file này

import cv2
import numpy as np
from .palette import COLOR_PALETTE_BGR


def _bgr_to_lab(bgr: tuple[int, int, int]) -> np.ndarray:
    bgr_arr = np.uint8([[list(bgr)]])
    lab = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return lab


# cache LAB palette để chạy realtime nhanh
_PALETTE_LAB = {name: _bgr_to_lab(bgr) for name, bgr in COLOR_PALETTE_BGR.items()}


def dominant_bgr_kmeans(
    bgr_img: np.ndarray,
    k: int = 3,
    sample: int = 6000,
    resize: int = 160,
) -> tuple[int, int, int]:
    """
    Lấy màu chủ đạo (BGR) bằng KMeans.
    Có lọc pixel để giảm nhiễu do highlight/trắng/da tay.
    """
    if bgr_img is None or bgr_img.size == 0:
        return (0, 0, 0)

    img = cv2.resize(bgr_img, (resize, resize), interpolation=cv2.INTER_AREA)

    # ===== lọc pixel để giảm nhiễu do highlight/trắng/da tay =====
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # giữ pixel "có màu" rõ hơn: tránh trắng/chói
    mask = (S > 35) & (V > 25) & (V < 245)
    pixels = img[mask]

    # nếu lọc quá mạnh (đen/xám/trắng) -> fallback toàn ảnh
    if pixels is None or len(pixels) < 300:
        pixels = img.reshape(-1, 3)

    # sample để nhanh
    if len(pixels) > sample:
        idx = np.random.choice(len(pixels), sample, replace=False)
        pixels = pixels[idx]

    Z = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())
    dom = centers[int(np.argmax(counts))]
    return (int(dom[0]), int(dom[1]), int(dom[2]))  # BGR


def nearest_color_name_lab(bgr: tuple[int, int, int], *, forbid_brown: bool = False) -> tuple[str, float]:
    """
    Tìm màu gần nhất trong palette bằng khoảng cách LAB.
    forbid_brown=True: cấm trả ra 'brown' (dùng khi màu đậm, S cao).
    """
    lab = _bgr_to_lab(bgr)
    best_name, best_d = "unknown", 1e9

    for name, ref_lab in _PALETTE_LAB.items():
        if forbid_brown and name == "brown":
            continue

        d = float(np.linalg.norm(lab - ref_lab))
        if d < best_d:
            best_d, best_name = d, name

    return best_name, best_d


def _hsv_medians_from_crop(crop_bgr: np.ndarray) -> tuple[float, float, float]:
    """
    Lấy median HSV của crop (giúp rule semantic ổn định hơn).
    """
    img = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    return float(np.median(H)), float(np.median(S)), float(np.median(V))


def detect_neutral_from_crop(crop_bgr: np.ndarray) -> str | None:
    """
    Detect black/white/gray bằng thống kê HSV (robust hơn LAB cho neutral).
    Đã tune để hộp trắng thường ra 'white' thay vì 'gray'.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    _, s_med, v_med = _hsv_medians_from_crop(crop_bgr)

    # Black: very low V
    if v_med < 55:
        return "black"

    # White: low S + fairly high V (tune thực tế)
    if s_med < 30 and v_med > 170:
        return "white"

    # Gray: low S + mid V
    if s_med < 30 and 55 <= v_med <= 170:
        return "gray"

    return None


def detect_light_color_from_crop(crop_bgr: np.ndarray) -> str | None:
    """
    Detect pastel/light colors: saturation thấp nhưng vẫn có hue.
    Giúp 'xanh pastel' không bị dính sang gray.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    h_med, s_med, v_med = _hsv_medians_from_crop(crop_bgr)

    # pastel/light: low S, đủ sáng
    if s_med < 40 and v_med > 120:
        # OpenCV Hue: 0-180
        if 90 <= h_med <= 130:
            return "light blue"
        if 40 <= h_med <= 80:
            return "light green"
        if 15 < h_med < 40:
            return "light yellow"
        if 0 <= h_med <= 15 or 165 <= h_med <= 180:
            return "light red"
        if 130 < h_med < 165:
            return "light pink"

    return None


def hue_priority_override(h_med: float, s_med: float) -> str | None:
    """
    Ưu tiên semantic cho nhóm đỏ/hồng/tím khi màu còn 'đậm' (S cao).
    Fix case 'pink' bị nhầm sang 'brown'.
    """
    if s_med < 50:
        return None  # chỉ áp dụng khi màu đủ đậm

    # Hue OpenCV: 0-180
    # vùng đỏ (wrap): [0..10] U [160..180]
    if 160 <= h_med <= 180 or 0 <= h_med <= 10:
        return "red"
    # vùng hồng/magenta
    if 130 <= h_med < 160:
        return "pink"
    # vùng tím
    if 110 <= h_med < 130:
        return "purple"

    return None

def detect_color_name_from_crop(
    crop_bgr: np.ndarray,
    unknown_dist_threshold: float = 55.0,
) -> tuple[str, tuple[int, int, int], float]:
    """
    FINAL COLOR LOGIC (STABLE):

    1) black / white
    2) hue-based colors (yellow, orange, green, blue, red, pink, purple)
    3) pastel / light
    4) gray
    5) LAB nearest (brown, fallback)
    """

    if crop_bgr is None or crop_bgr.size == 0:
        return ("unknown", (0, 0, 0), 999.0)

    h_med, s_med, v_med = _hsv_medians_from_crop(crop_bgr)

    # ===== 1. BLACK / WHITE =====
    if v_med < 55:
        dom = dominant_bgr_kmeans(crop_bgr, k=2)
        return ("black", dom, 0.0)

    if s_med < 30 and v_med > 170:
        dom = dominant_bgr_kmeans(crop_bgr, k=2)
        return ("white", dom, 0.0)

    # ===== 2. HUE-BASED COLORS (CORE FIX) =====
    if s_med >= 25:  # cho phép nilon / phản sáng nhẹ
        # OpenCV Hue range: 0–180

        # RED
        if h_med <= 10 or h_med >= 160:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("red", dom, 0.0)

        # ORANGE
        if 10 < h_med < 20:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("orange", dom, 0.0)

        # YELLOW  ⭐ FIX CASE CỦA BẠN
        if 20 <= h_med <= 40:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("yellow", dom, 0.0)

        # GREEN
        if 40 < h_med <= 85:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("green", dom, 0.0)

        # BLUE
        if 85 < h_med <= 120:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("blue", dom, 0.0)

        # PURPLE / PINK
        if 120 < h_med < 160:
            dom = dominant_bgr_kmeans(crop_bgr, k=3)
            return ("pink", dom, 0.0)

    # ===== 3. LIGHT / PASTEL =====
    light = detect_light_color_from_crop(crop_bgr)
    if light is not None:
        dom = dominant_bgr_kmeans(crop_bgr, k=2)
        return (light, dom, 0.0)

    # ===== 4. GRAY =====
    if s_med < 30:
        dom = dominant_bgr_kmeans(crop_bgr, k=2)
        return ("gray", dom, 0.0)

    # ===== 5. LAB NEAREST (BROWN / FALLBACK) =====
    dom_bgr = dominant_bgr_kmeans(crop_bgr, k=3)

    forbid_brown = (s_med >= 40)
    name, dist = nearest_color_name_lab(dom_bgr, forbid_brown=forbid_brown)

    if dist > unknown_dist_threshold:
        return ("unknown", dom_bgr, dist)

    return (name, dom_bgr, dist)
