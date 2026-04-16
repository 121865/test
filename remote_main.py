import pyzed.sl as sl
from ultralytics import YOLO
import cv2
import socket
import time
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
    
    # 優化連線參數 (移除報錯的 open_timeout_msec)
    init_params.sdk_verbose = True
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # 筆電端不運算深度，節省效能
    init_params.camera_resolution = sl.RESOLUTION.VGA
    
    print(f"正在嘗試連線至遠端串流: {tx2_ip}:{stream_port}...")
    
    # 嘗試開啟相機
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"\n連線失敗: {status}")
        print("請確認：")
        print("1. TX2 端是否已跑起 tx2_native_aa.py 且畫面停在 bBlitMode...")
        print("2. 筆電 Ping 100.73.177.103 是否穩定 (需在 100ms 內)")
        print("3. TX2 端是否執行了 sudo ufw disable")
        return

    # 3. 載入 YOLO
    print("連線成功！正在載入 YOLO 模型...")
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
    print("按 'o' (英文小寫): 開啟避障")
    print("按 'p' (英文小寫): 關閉避障")
    print("按 'q' (英文小寫): 退出程式\n")

    while True:
        # 抓取影像
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            
            # 轉換為 BGR (OpenCV 格式)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # YOLO 辨識
            results = model.predict(frame, conf=0.25, verbose=False)[0]
            annotated_frame = results.plot()
            
            # 顯示避障狀態提示
            color = (0, 255, 0) if oa_status == "ON" else (0, 0, 255)
            cv2.putText(annotated_frame, f"Obstacle Avoidance: {oa_status}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Boat Remote Master Control", annotated_frame)
            
        # 監聽鍵盤
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o'):
            # 連發 5 次，確保 TX2 收到
            for _ in range(3):
                cmd_sock.sendto(b"OA_ON", (tx2_ip, cmd_port))
                time.sleep(0.01) # 微小間隔避免阻塞
            oa_status = "ON"
            print(">>> 指令已送出：開啟避障 (連發 5 次模式)")
            
        elif key == ord('p'):
            # 連發 5 次，確保 TX2 收到
            for _ in range(3):
                cmd_sock.sendto(b"OA_OFF", (tx2_ip, cmd_port))
                time.sleep(0.01)
            oa_status = "OFF"
            print(">>> 指令已送出：關閉避障 (連發 5 次模式)")

    # 釋放資源
    zed.close()
    cv2.destroyAllWindows()
    print("程式已結束")

if __name__ == "__main__":
    main()