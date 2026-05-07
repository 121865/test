import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
from datetime import datetime

def main():
    # 1. 設定 TX2 連線資訊
    tx2_ip = "100.73.177.103"
    stream_port = 30000
    cmd_port = 50005
    
    # 初始化 UDP 指令發送器 (用於控制避障開關)
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 2. 初始化 ZED 相機連線
    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # 設定連線來源為遠端串流
    init_params.set_from_stream(tx2_ip, stream_port)
    init_params.sdk_verbose = True
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 筆電端不運算深度，節省效能
    init_params.camera_resolution = sl.RESOLUTION.VGA
    
    print(f"正在嘗試連線至遠端串流: {tx2_ip}:{stream_port}...")
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"\n連線失敗: {status}")
        return

    # 3. 設定影片錄製器 (VideoWriter) - 用於錄製「無框」原始影像
    # 檔名包含日期時間，避免覆蓋
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_raw_data.mp4")
    # ZED VGA 解析度為 640x480
    frame_size = (640, 480)
    fps = 15 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    print(f">>> 錄影已啟動，原始影像將儲存至: {filename}")

    # 4. 載入 YOLO
    print("正在載入 YOLO 模型...")
    try:
        model = YOLO('best.pt') 
    except Exception as e:
        print(f"模型載入失敗: {e}")
        zed.close()
        return

    image = sl.Mat()
    runtime = sl.RuntimeParameters()
    oa_status = "OFF" 

    print("\n操作說明:")
    print("按 'o': 開啟避障 | 按 'p': 關閉避障 | 按 'q': 停止錄影並退出\n")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # 獲取左眼影像
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # --- 色調修正關鍵步驟 ---
            # 1. get_data() 拿到的是 RGBA (4通道)
            raw_rgba = image.get_data()
            # 2. 轉換為 BGR (OpenCV 標準 3通道)，這能解決色調不正常(如藍色變橘色)的問題
            frame_raw = cv2.cvtColor(raw_rgba, cv2.COLOR_RGBA2BGR)
            
            # --- 錄影儲存 (存下還沒畫框的原始馬賽克影像) ---
            video_writer.write(frame_raw)
            
            # 3. YOLO 辨識 (在複製品上畫框，不影響存檔)
            results = model.predict(frame_raw, conf=0.25, verbose=False)[0]
            annotated_frame = results.plot()
            
            # 顯示 UI 提示
            color = (0, 255, 0) if oa_status == "ON" else (0, 0, 255)
            cv2.putText(annotated_frame, f"Avoidance: {oa_status} | REC", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Boat Remote Master Control", annotated_frame)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            for _ in range(5): # 增加連發次數確保穩定
                cmd_sock.sendto(b"OA_ON", (tx2_ip, cmd_port))
                time.sleep(0.01)
            oa_status = "ON"
            print(">>> 指令：開啟避障")
        elif key == ord('p'):
            for _ in range(5):
                cmd_sock.sendto(b"OA_OFF", (tx2_ip, cmd_port))
                time.sleep(0.01)
            oa_status = "OFF"
            print(">>> 指令：關閉避障")

    # 釋放資源
    video_writer.release() # 務必 release 否則影片會毀損
    zed.close()
    cv2.destroyAllWindows()
    print(f"\n錄影結束，原始素材已存於 {filename}")

if __name__ == "__main__":
    main()