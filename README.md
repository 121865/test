1. 
船端 (Edge Computer): NVIDIA Jetson TX2 

作業系統: Ubuntu 18.04 。

開發語言: Python 3.8 (虛擬環境 venv38)。


核心軟體: ZED SDK (負責影像編碼與深度運算) 。


通訊工具: Tailscale (負責提供虛擬 IP: 100.73.177.103) 。

2. 岸端 (Master Control): 筆電
開發語言: Python (虛擬環境 zed_api_env)。

核心軟體: YOLOv8/v10 (負責影像辨識)、OpenCV (負責顯示 UI)。

角色: 負責高負載的 AI 運算，並遠端控制避障開關。

🚀 完整操作流程
請務必依照以下順序執行，以確保 ZED 相機與網路連線能順利握手：

步驟一：建立網路隧道 (Tailscale)
確保 TX2 與筆電均已啟動 Tailscale 並登入同一帳號。

在筆電上執行 ping 100.73.177.103，確認延遲在穩定範圍內。

步驟二：啟動船端核心 (tx2_native_aa.py)
在 TX2 的終端機執行：

```Bash
cd ~/Desktop/zed_ship_project/  
source venv38/bin/activate  
python tx2_native_aa.py
```
運行狀態: 當畫面上出現 NVMEDIA_ENC: bBlitMode is set to TRUE 時，代表 H.265 硬體壓縮串流已開始在 Port 30000 廣播。

監聽狀態: 腳本會同步開啟 Thread 監聽 Port 50005 的 UDP 控制訊號。

步驟三：啟動岸端監控 (remote_main.py)
在筆電的終端機執行：

PowerShell
python remote_main.py
連線過程: 筆電會透過 ZED SDK 的 set_from_stream 函式抓取 TX2 的影像。

AI 推論: 畫面跳出後，YOLO 會即時辨識目標船隻，並在視窗中顯示偵測結果。

步驟四：即時避障控制與交互
當影像視窗出現後，點擊視窗使其獲得焦點，即可使用鍵盤進行控制：

按 o (開啟避障): 筆電會連續發送 3 次 OA_ON 訊號給 TX2。TX2 接收後啟動 ROI 深度計算，並在終端機顯示前方距離（如：✅ 安全中: 3.50m）。

按 p (關閉避障): 筆電發送 OA_OFF。TX2 停止深度運算，將所有算力保留給影像串流，以提升畫面流暢度。
