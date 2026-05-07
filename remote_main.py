import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
from datetime import datetime

def main():
    tx2_ip = "100.73.177.103"
    stream_port = 30000
    cmd_port = 50005
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_stream(tx2_ip, stream_port)
    init_params.depth_mode = sl.DEPTH_MODE.NONE  
    init_params.camera_resolution = sl.RESOLUTION.VGA 
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"連線失敗: {status}")
        return

    # 錄影設定 (使用 XVID 提高相容性)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_raw_data.avi")
    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video_writer = cv2.VideoWriter(filename, fourcc, 15.0, frame_size)
    
    try:
        print(f"正在錄製原始影像至: {filename}")
        model = YOLO('best.pt') 
        image = sl.Mat()
        runtime = sl.RuntimeParameters()
        oa_status = "OFF" 

        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                
                # --- 色調修正關鍵：ZED 預設多為 BGRA 排列 ---
                raw_rgba = image.get_data()
                # 這裡改用 BGRA -> BGR
                frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_BGRA2BGR)
                
                # 同步儲存原始素材 (無框)
                video_writer.write(frame_raw)
                
                # YOLO 辨識
                results = model.predict(frame_raw, conf=0.25, verbose=False)[0]
                annotated_frame = results.plot()
                
                # 介面提示
                color = (0, 255, 0) if oa_status == "ON" else (0, 0, 255)
                cv2.putText(annotated_frame, f"Avoidance: {oa_status} | REC", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Boat Control Console", annotated_frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
        print("正在關閉並封裝影片...")
        video_writer.release() 
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()