import cv2 as cv
import matplotlib.pyplot as plt
import halftone  # 引入你寫的 halftone.py

if __name__ == "__main__":
    # 讀取灰階圖像
    img = cv.imread("Baboon-image.png", cv.IMREAD_GRAYSCALE)

    # 產生 Bayer matrix 和 threshold 矩陣
    n = 2  # 產生 4x4 Bayer 矩陣
    bayer = halftone.generate_bayer_matrix(n)
    thresholds = halftone.generate_thresholds_matrix(bayer)

    # 做 Ordered Dithering
    od_img = halftone.Ordered_Dithering(img, thresholds)

    # 顯示結果 (用 matplotlib)
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Ordered Dithering")
    plt.imshow(od_img, cmap="gray")

    plt.show()