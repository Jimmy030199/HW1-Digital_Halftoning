import sys
import cv2 as cv
import numpy as np
import matplotlib as plt

# 建立 Bayer 矩陣
def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller_matrix = generate_bayer_matrix(n - 1)
        size = 2 ** n
        new_matrix = np.zeros((size, size), dtype=int)
        for i in range(2 ** (n - 1)):
            for j in range(2 ** (n - 1)):
                base_value = 4 * smaller_matrix[i, j]
                new_matrix[i, j] = base_value
                new_matrix[i, j + 2 ** (n - 1)] = base_value + 2
                new_matrix[i + 2 ** (n - 1), j] = base_value + 3
                new_matrix[i + 2 ** (n - 1), j + 2 ** (n - 1)] = base_value + 1
        
        return new_matrix

def generate_thresholds_matrix(bayer_matrix):
    N = bayer_matrix.shape[0]
    thresholds_matrix = np.zeros_like(bayer_matrix, int)

    # TODO:Calculate each bayer matrix element threshold

    # 使用公式計算 threshold
    thresholds_matrix = ((bayer_matrix + 0.5) / (N * N)) * 255 

    # 把結果轉換成 8 位元整數 (0~255)，符合 OpenCV 的灰階影像格式 
    thresholds_matrix = thresholds_matrix.astype(np.uint8)
    return thresholds_matrix

def Ordered_Dithering(img, thresholds_matrix):
    # TODO:Implementing the ordered dithering algorithm

    # 取得閾值矩陣大小 (例如 4x4 → N=4)
    N = thresholds_matrix.shape[0]

    # 建立輸出的空影像 (跟輸入一樣大小)
    Ordered_Dithering_img = np.zeros_like(img, dtype=np.uint8)

    # 逐像素套用閾值
    for i in range(img.shape[0]):      # 高度
        for j in range(img.shape[1]):  # 寬度
            threshold = thresholds_matrix[i % N, j % N]   # 對應的閾值
            if img[i, j] > threshold:
                Ordered_Dithering_img[i, j] = 255
            else:
                Ordered_Dithering_img[i, j] = 0

    return Ordered_Dithering_img


def Error_Diffusion(img):
    # TODO:Implementing the error diffusion algorithm

    return Error_Diffusion_img

if __name__ == '__main__':

    img = cv.imread(sys.argv[1])

    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)
    # TODO:Show your picture

