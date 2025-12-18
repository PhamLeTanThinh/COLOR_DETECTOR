import cv2
import numpy as np
from ultralytics import YOLO
from color import detect_color_name_from_crop

MODEL_PATH = "yolov8n.pt"   # <<< QUAN TRỌNG: quay về n
PERSON_CLASS_ID = 0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crop_with_padding(img, x1, y1, x2, y2, pad=10):
    h, w = img.shape[:2]
    x1 = clamp(int(x1) - pad, 0, w - 1)
    y1 = clamp(int(y1) - pad, 0, h - 1)
    x2 = clamp(int(x2) + pad, 0, w - 1)
    y2 = clamp(int(y2) + pad, 0, h - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def center_crop(img, ratio=0.6):
    h, w = img.shape[:2]
    nh, nw = int(h * ratio), int(w * ratio)
    y1 = (h - nh) // 2
    x1 = (w - nw) // 2
    return img[y1:y1+nh, x1:x1+nw].copy()


def smooth_bbox(prev, curr, alpha=0.85):
    if prev is None:
        return curr
    return (
        int(alpha * prev[0] + (1 - alpha) * curr[0]),
        int(alpha * prev[1] + (1 - alpha) * curr[1]),
        int(alpha * prev[2] + (1 - alpha) * curr[2]),
        int(alpha * prev[3] + (1 - alpha) * curr[3]),
    )


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_thres = 0.20
    pad = 12

    phone_box_ratio = 0.34
    min_area_ratio = 0.05
    max_area_ratio = 4.0

    prev_box = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ROI square
        side = int(min(w, h) * phone_box_ratio)
        cx, cy = w // 2, h // 2
        rx1, ry1 = cx - side // 2, cy - side // 2
        rx2, ry2 = cx + side // 2, cy + side // 2
        roi_box = (rx1, ry1, rx2, ry2)
        roi_area = side * side

        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
        cv2.putText(frame, "Put object in square (press q to quit)",
                    (rx1, ry1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # === detect mỗi 2 frame ===
        detect_now = (frame_idx % 2 == 0)

        if detect_now:
            results = model.predict(frame, conf=conf_thres, verbose=False)
            r = results[0]

            best = None
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0].item())
                    if cls_id == PERSON_CLASS_ID:
                        continue

                    conf = float(b.conf[0].item())
                    x1, y1, x2, y2 = b.xyxy[0].tolist()

                    cx_box = (x1 + x2) / 2
                    cy_box = (y1 + y2) / 2
                    if not (rx1 <= cx_box <= rx2 and ry1 <= cy_box <= ry2):
                        continue

                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    area_ratio = area / (roi_area + 1e-6)
                    if not (min_area_ratio <= area_ratio <= max_area_ratio):
                        continue

                    if best is None or conf > best[0]:
                        best = (conf, x1, y1, x2, y2)

            if best is not None:
                _, x1, y1, x2, y2 = best
                prev_box = smooth_bbox(prev_box, (int(x1), int(y1), int(x2), int(y2)))

        # === dùng bbox đã smooth ===
        if prev_box is not None:
            x1, y1, x2, y2 = prev_box
            crop = crop_with_padding(frame, x1, y1, x2, y2, pad=pad)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 5, (0,0,255), -1)

            if crop is not None:
                crop_center = center_crop(crop, ratio=0.6)
                color_name, dom_bgr, dist = detect_color_name_from_crop(
                    crop_center, unknown_dist_threshold=45.0
                )

                cv2.putText(frame, f"Color: {color_name}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("YOLO -> Color Name", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
