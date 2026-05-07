import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
from datetime import datetime

def main():
    tx2_ip = "100.73.177.103"
    stream_port = 30000
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_stream(tx2_ip, stream_port)
    init_params.camera_resolution = sl.RESOLUTION.VGA 
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("連線失敗")
        return

    # 先宣告 video_writer 為 None，等抓到第一張圖再初始化
    video_writer = None
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_raw_data.avi")
    
    try:
        model = YOLO('best.pt') 
        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        frame_count = 0

        print(">>> 正在等待影像串流...")

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                raw_rgba = image.get_data()
                
                # 色調修正 (BGRA to BGR)
                frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_BGRA2BGR)
                h, w, _ = frame_raw.shape

                # --- 自動初始化錄影機 (動態適配尺寸) ---
                if video_writer is None:
                    print(f">>> 偵測到影像尺寸: {w}x{h}，開始錄影...")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(filename, fourcc, 15.0, (w, h))

                # 寫入影片並計數
                video_writer.write(frame_raw)
                frame_count += 1
                
                # YOLO 辨識與顯示
                results = model.predict(frame_raw, conf=0.25, verbose=False)[0]
                annotated_frame = results.plot()
                
                cv2.putText(annotated_frame, f"REC Frames: {frame_count} | {w}x{h}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Master Control", annotated_frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if video_writer:
            video_writer.release()
        zed.close()
        cv2.destroyAllWindows()
        print(f"\n錄影結束，共計 {frame_count} 幀，存於 {filename}")

if __name__ == "__main__":
    main()