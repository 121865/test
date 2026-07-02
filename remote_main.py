import pyzed.sl as sl
from ultralytics import YOLO
import cv2
from datetime import datetime
from pathlib import Path

# =========================
# 可調參數
# =========================
TX2_IP = "100.73.177.103"
STREAM_PORT = 30000

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best.pt")
TRACKER_CFG = "bytetrack.yaml"

CONF_THRES = 0.30
IMGSZ = 640

ONLY_SHIP = True
MIN_BOX_AREA = 0   # 可改成 200 / 300 過濾太小假框

OUTPUT_DIR = Path(r"D:\SeaShips_train\result")


def xyxy_to_int(xyxy):
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)


def make_video_writer(path, w, h, fps=15.0):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(str(path), fourcc, fps, (w, h))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_stream(TX2_IP, STREAM_PORT)
    init_params.camera_resolution = sl.RESOLUTION.VGA

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("連線失敗")
        return

    model = YOLO(MODEL_PATH)
    names = model.names

    ship_ids = []
    if ONLY_SHIP:
        ship_ids = [k for k, v in names.items() if str(v).lower() == "ship"]

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    raw_writer = None
    track_writer = None
    raw_recording = False
    track_recording = False

    frame_count = 0

    print(">>> 正在等待影像串流...")
    print(f">>> Model: {MODEL_PATH}")
    print(f">>> Tracker: {TRACKER_CFG}")
    print(">>> 按鍵功能：")
    print("    q = 離開")
    print("    r = 開始/停止錄製原始影像")
    print("    t = 開始/停止錄製追蹤影像")
    print("    s = 儲存目前畫面(raw + track)")

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                raw_rgba = image.get_data()

                # BGRA -> BGR
                frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_BGRA2BGR)
                h, w, _ = frame_raw.shape
                frame_count += 1

                # =========================
                # B3 + ByteTrack
                # =========================
                results = model.track(
                    source=frame_raw,
                    persist=True,
                    tracker=TRACKER_CFG,
                    conf=CONF_THRES,
                    imgsz=IMGSZ,
                    verbose=False,
                    device=0,
                )

                vis = frame_raw.copy()

                if results and len(results) > 0:
                    r = results[0]

                    if r.boxes is not None and len(r.boxes) > 0:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        boxes_cls = r.boxes.cls.cpu().numpy().astype(int)
                        boxes_conf = r.boxes.conf.cpu().numpy()

                        track_ids = None
                        if r.boxes.id is not None:
                            track_ids = r.boxes.id.cpu().numpy().astype(int)

                        for i in range(len(boxes_xyxy)):
                            cls_id = boxes_cls[i]
                            conf = float(boxes_conf[i])

                            if ship_ids and cls_id not in ship_ids:
                                continue

                            x1, y1, x2, y2 = xyxy_to_int(boxes_xyxy[i])
                            area = max(0, x2 - x1) * max(0, y2 - y1)
                            if area < MIN_BOX_AREA:
                                continue

                            track_id = -1
                            if track_ids is not None and i < len(track_ids):
                                track_id = int(track_ids[i])

                            cls_name = names.get(cls_id, str(cls_id))
                            if track_id >= 0:
                                label = f"ID {track_id} | {cls_name} {conf:.2f}"
                            else:
                                label = f"{cls_name} {conf:.2f}"

                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                vis,
                                label,
                                (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

                # 狀態顯示
                status_text = f"Frames: {frame_count} | {w}x{h} | RAW_REC: {'ON' if raw_recording else 'OFF'} | TRACK_REC: {'ON' if track_recording else 'OFF'}"
                cv2.putText(
                    vis,
                    status_text,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                # 如果啟用錄影，就寫入
                if raw_recording and raw_writer is not None:
                    raw_writer.write(frame_raw)

                if track_recording and track_writer is not None:
                    track_writer.write(vis)

                cv2.imshow("Master Control - B3 + ByteTrack", vis)

            key = cv2.waitKey(1) & 0xFF

            # q: 離開
            if key == ord('q'):
                break

            # r: 開/關 raw 錄影
            elif key == ord('r'):
                if not raw_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_filename = OUTPUT_DIR / f"{timestamp}_raw_data.avi"
                    raw_writer = make_video_writer(raw_filename, w, h, fps=15.0)
                    raw_recording = True
                    print(f">>> 開始錄製原始影像: {raw_filename}")
                else:
                    raw_recording = False
                    if raw_writer is not None:
                        raw_writer.release()
                        raw_writer = None
                    print(">>> 停止錄製原始影像")

            # t: 開/關 track 錄影
            elif key == ord('t'):
                if not track_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    track_filename = OUTPUT_DIR / f"{timestamp}_b3_bytetrack.avi"
                    track_writer = make_video_writer(track_filename, w, h, fps=15.0)
                    track_recording = True
                    print(f">>> 開始錄製追蹤影像: {track_filename}")
                else:
                    track_recording = False
                    if track_writer is not None:
                        track_writer.release()
                        track_writer = None
                    print(">>> 停止錄製追蹤影像")

            # s: 存單張圖片
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                raw_img_path = OUTPUT_DIR / f"{timestamp}_raw.jpg"
                track_img_path = OUTPUT_DIR / f"{timestamp}_track.jpg"

                cv2.imwrite(str(raw_img_path), frame_raw)
                cv2.imwrite(str(track_img_path), vis)
                print(f">>> 已儲存畫面:\n    RAW   : {raw_img_path}\n    TRACK : {track_img_path}")

    finally:
        if raw_writer is not None:
            raw_writer.release()
        if track_writer is not None:
            track_writer.release()

        zed.close()
        cv2.destroyAllWindows()
        print(f"\n結束，共計 {frame_count} 幀")


if __name__ == "__main__":
    main()