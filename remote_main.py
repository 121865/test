import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
from datetime import datetime
import os

def main():
    # 1. 連線設定
    tx2_ip = "100.73.177.103"
    stream_port = 30000
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_stream(tx2_ip, stream_port)
    init_params.camera_resolution = sl.RESOLUTION.VGA # 強制 VGA
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("連線失敗")
        return

    # 2. 影片錄製設定 (使用最穩定的 XVID + .avi)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    # 這裡務必與 ZED 輸出的 640x480 一致
    video_writer = cv2.VideoWriter(filename, fourcc, 15.0, (640, 480))
    
    print(f">>> 準備錄影至: {os.path.abspath(filename)}")

    try:
        model = YOLO('best.pt') 
        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        frame_count = 0

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                
                # 取得資料並轉為 BGR
                raw_rgba = image.get_data()
                frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_BGRA2BGR)
                
                # 檢查影像大小是否為 640x480
                if frame_raw.shape[1] == 640 and frame_raw.shape[0] == 480:
                    video_writer.write(frame_raw)
                    frame_count += 1
                
                # YOLO 辨識
                results = model.predict(frame_raw, conf=0.25, verbose=False)[0]
                annotated_frame = results.plot()
                
                # 在畫面上顯示已錄製幀數，方便妳即時確認有沒有在存檔
                cv2.putText(annotated_frame, f"REC Frames: {frame_count}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Master Control", annotated_frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_writer.release()
        zed.close()
        cv2.destroyAllWindows()
        print(f"\n錄影結束，總共錄製了 {frame_count} 幀。")
        if frame_count == 0:
            print("警告：錄製幀數為 0，請檢查 ZED 解析度設定。")

if __name__ == "__main__":
    main()