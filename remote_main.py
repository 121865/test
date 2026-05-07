import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
from datetime import datetime

def main():
    # 1. 連線資訊設定
    tx2_ip = "100.73.177.103"
    stream_port = 30000
    cmd_port = 50005
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 2. ZED 相機初始化
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_stream(tx2_ip, stream_port)
    init_params.depth_mode = sl.DEPTH_MODE.NONE  
    init_params.camera_resolution = sl.RESOLUTION.VGA 
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"連線失敗: {status}")
        return

    # 3. 影片錄製器設定 (使用 MJPG 編碼，這是 .avi 最穩定的格式)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_raw_data.avi")
    frame_size = (640, 480)
    fps = 15.0
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    print(f">>> 錄影已啟動：{filename}")

    try:
        model = YOLO('best.pt') 
        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        oa_status = "OFF" 

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                
                # --- 色調修正：BGR 轉換 ---
                raw_rgba = image.get_data()
                frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_BGRA2BGR)
                
                # 寫入影片 (無辨識框原始畫面) 
                video_writer.write(frame_raw)
                
                # YOLO 辨識與顯示
                results = model.predict(frame_raw, conf=0.25, verbose=False)[0]
                annotated_frame = results.plot()
                
                # 顯示 UI
                color = (0, 255, 0) if oa_status == "ON" else (0, 0, 255)
                cv2.putText(annotated_frame, f"Avoidance: {oa_status} | REC", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Boat Remote Master Control", annotated_frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # 按 'q' 安全退出 
                break
            elif key == ord('o'):
                for _ in range(5): cmd_sock.sendto(b"OA_ON", (tx2_ip, cmd_port))
                oa_status = "ON"
            elif key == ord('p'):
                for _ in range(5): cmd_sock.sendto(b"OA_OFF", (tx2_ip, cmd_port))
                oa_status = "OFF"

    except Exception as e:
        print(f"發生錯誤: {e}")
    finally:
        # --- 安全關閉與檔案封口  ---
        print("正在安全關閉並儲存影片...")
        video_writer.release() 
        zed.close()
        cv2.destroyAllWindows()
        print(f"影片儲存完成: {filename}")

if __name__ == "__main__":
    main()