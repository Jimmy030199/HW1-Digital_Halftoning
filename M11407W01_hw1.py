import cv2 as cv
import matplotlib.pyplot as plt
from halftone import generate_bayer_matrix, generate_thresholds_matrix,Ordered_Dithering,Error_Diffusion

if __name__ == "__main__":
    # 讀取灰階圖像(會把圖片 轉成灰階 (單通道) 讀進來。)
    img = cv.imread("images/Baboon-image.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("images/F-16-image.png", cv.IMREAD_GRAYSCALE)


    # 產生 Bayer matrix 和 threshold 矩陣
    n = 2  # 產生 4x4 Bayer 矩陣
    bayer = generate_bayer_matrix(n)
    thresholds = generate_thresholds_matrix(bayer)

    # 做 Ordered Dithering
    od_img = Ordered_Dithering(img, thresholds)
    od_img2 = Ordered_Dithering(img2, thresholds)


    # 做 Error_Diffusion
    ed_img=Error_Diffusion(img)
    ed_img2=Error_Diffusion(img2)


 

    # 原始圖片
    plt.subplot(2,3,1)
    plt.title("origin Baboon", fontsize=10)
    plt.imshow(img, cmap="gray")
    plt.axis("off")   # 不顯示座標軸

    plt.subplot(2,3,4)
    plt.title("origin F-16", fontsize=10)
    plt.imshow(img2, cmap="gray")
    plt.axis("off")   # 不顯示座標軸

    # Ordered Dithering處理後圖片
    plt.subplot(2,3,2)
    plt.title("Ordered Dithering ", fontsize=10)
    plt.imshow(od_img, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.title("Ordered Dithering ", fontsize=10)
    plt.imshow(od_img2, cmap="gray")
    plt.axis("off")

    # Error_Diffusion處理後圖片
    plt.subplot(2,3,3)
    plt.title("Error_Diffusion ", fontsize=10)
    plt.imshow(ed_img, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.title("Error_Diffusion ", fontsize=10)
    plt.imshow(ed_img2, cmap="gray")
    plt.axis("off")

  

    plt.show()


    