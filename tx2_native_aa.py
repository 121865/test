import pyzed.sl as sl
import cv2
import numpy as np
import socket
import threading

oa_enabled = False

def command_listener():
    global oa_enabled
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("",50005))
    print("Port50005...")
    
    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode().strip()
        if msg == "OA_ON":
            oa_enabled = True
            print("\n>>> on! ")
        elif msg == "OA_OFF":
            oa_enabled = False
            print("\n>>> off!")

def main():
    threading.Thread(target=command_listener, daemon=True).start()

    # 1. 初始化 ZED 相機
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA    # 低解析度節省算力
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA         # 避障需要較準確的深度
    init_params.coordinate_units = sl.UNIT.METER         # 使用公尺為單位
    init_params.camera_fps = 10
    
    # 2. 開啟串流功能 (讓筆電端可以連線)
    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H265
    stream_params.bitrate = 100                          # 針對妳的網路環境設為 100kbps
    stream_params.port = 30000
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("相機開啟失敗")
        return

    # 啟動對外串流
    zed.enable_streaming(stream_params)
    
    # 3. 準備物件
    depth_map = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    print("\n" + "="*30)
    print("TX2 control system on...")
    print("streaming port : 30000 | port: 50005")
    print("="*30)

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                if oa_enabled:
                    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                    depth_data = depth_map.get_data()
                    h, w = depth_data.shape
                    roi = depth_data[int(h*0.4):int(h*0.6), int(w*0.3):int(w*0.7)]
                    avg_dist = np.nanmean(roi)
                    
                    if avg_dist < 2.0:
                        print(f"警告！前方障礙物距離: {avg_dist:.2f}m -> 執行避讓動作", end="\r")
                    else:
                        print(f"偵測中: {avg_dist:.2f}m    ",end="\r")
            else:
                print("stop...", end="\r")
                
            
    except KeyboardInterrupt:
        print("\n正在關閉系統...")
    finally:
        zed.disable_streaming()
        zed.close()

if __name__ == "__main__":
    main()
