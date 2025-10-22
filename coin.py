import cv2
import numpy as np

COIN_SIZES = {
    "50NTD": (81, 100),
    "10NTD": (68, 80),
    "5NTD": (64, 67),
    "1NTD": (35, 63)
}
# -----------------------------------------------------------------------------------------

# 為了計算總金額，我們需要一個面額到數值的映射
DENOMINATION_VALUES = {
    "50NTD": 50,
    "10NTD": 10,
    "5NTD": 5,
    "1NTD": 1,
    "未知": 0
}

def recognize_coin(radius):
    for denomination, (min_r, max_r) in COIN_SIZES.items():
        if min_r <= radius <= max_r:
            return denomination
    return "未知"


def coin_detector(image_path, output_path='output.jpg'):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'. Check the file path and name.")
        return

    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Preprocessing: Gaussian Blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 3. Circle Detection: Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,           # Inverse ratio of the accumulator resolution
        minDist=60,       # Minimum distance between detected centers (pixels)
        param1=50,        # Higher threshold for the Canny edge detector
        param2=45,        # Accumulator threshold
        minRadius=30,     # Min radius (pixels) to search for
        maxRadius=110     # Max radius (pixels) to search for
    )

    total_coins = 0
    coin_counts = {"50NTD": 0, "10NTD": 0, "5NTD": 0, "1NTD": 0, "未知": 0}

    # 4. Process detection results and draw on the image
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for (x, y, r) in circles[0, :]:
            # Identify denomination
            denomination = recognize_coin(r)

            # Draw outer circle (Green)
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            # Draw center (Red)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

            # Display denomination and radius (Blue text)
            text = f"{denomination} (r={r})"
            cv2.putText(output, text, (x - r, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Statistics
            coin_counts[denomination] += 1
            total_coins += 1

    # 5. Output results and **CALCULATE TOTAL AMOUNT**
    total_amount = 0
    recognized_total = 0
    
    # 計算總金額
    for denom, count in coin_counts.items():
        amount = count * DENOMINATION_VALUES.get(denom, 0)
        total_amount += amount
        if denom != "未知":
            recognized_total += count

    print("================== 硬幣辨識結果 (Coin Recognition Results) ==================")
    
    # 輸出總硬幣數量
    print(f"硬幣總共有：{total_coins} 個。") 
    
    print("\n各面額數量統計 (Denomination Counts)：")
    for denom, count in coin_counts.items():
        if denom != "未知":
            print(f"  {denom}: {count} 個 (總值: {count * DENOMINATION_VALUES[denom]} NTD)")
    
    # 輸出未識別的數量
    if coin_counts["未知"] > 0:
        print(f"  未知 (Unknown): {coin_counts['未知']} 個 (未計入總金額)")

    # 輸出總金額
    print(f"\n成功辨識面額的硬幣總數: {recognized_total} 個")
    print(f"💰 總金額 (Total Amount): {total_amount} NTD")
    print("================================================")

    # 6. Save the annotated image
    cv2.imwrite(output_path, output)
    print(f"處理後的圖片已保存為 '{output_path}'")

    # 7. Display the image (optional but recommended for visual check)
    cv2.imshow('Detected Coins', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Execute the function ---
# Ensure 'input.jpg' is in the same directory as the script.
coin_detector('input.jpg', 'output.jpg')