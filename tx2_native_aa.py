import pyzed.sl as sl
import numpy as np
import socket
import threading

oa_enabled = False


def command_listener():
    global oa_enabled
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", 50005))
    print("Port 50005 listening...")

    while True:
        data, _ = sock.recvfrom(1024)
        msg = data.decode().strip()
        if msg == "OA_ON":
            oa_enabled = True
            print("\n>>> OA ON")
        elif msg == "OA_OFF":
            oa_enabled = False
            print("\n>>> OA OFF")


def main():
    threading.Thread(target=command_listener, daemon=True).start()

    # =========================
    # 1. 初始化 ZED 相機
    # =========================
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.METER
    init_params.camera_fps = 10   # 比原本 10 稍高，追蹤會更穩

    # =========================
    # 2. 開啟串流功能
    # =========================
    stream_params = sl.StreamingParameters()
    stream_params.codec = sl.STREAMING_CODEC.H265
    stream_params.bitrate = 800   # 原本 100 太低，辨識追蹤容易糊掉
    stream_params.port = 30000

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("相機開啟失敗")
        return

    if zed.enable_streaming(stream_params) != sl.ERROR_CODE.SUCCESS:
        print("串流啟動失敗")
        zed.close()
        return

    depth_map = sl.Mat()
    runtime_params = sl.RuntimeParameters()

    print("\n" + "=" * 40)
    print("TX2 control system on...")
    print("Streaming port : 30000")
    print("Command port   : 50005")
    print("Resolution     : VGA")
    print("FPS            : 15")
    print("Bitrate        : 4000 kbps")
    print("=" * 40)

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                if oa_enabled:
                    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                    depth_data = depth_map.get_data()

                    h, w = depth_data.shape
                    roi = depth_data[int(h * 0.4):int(h * 0.6), int(w * 0.3):int(w * 0.7)]
                    avg_dist = np.nanmean(roi)

                    if avg_dist < 2.0:
                        print(f"警告！前方障礙物距離: {avg_dist:.2f} m -> 執行避讓動作", end="\r")
                    else:
                        print(f"偵測中: {avg_dist:.2f} m    ", end="\r")
            else:
                print("grab fail...", end="\r")

    except KeyboardInterrupt:
        print("\n正在關閉系統...")

    finally:
        zed.disable_streaming()
        zed.close()
        print("TX2 system closed.")


if __name__ == "__main__":
    main()