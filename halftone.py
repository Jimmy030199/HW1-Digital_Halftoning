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

    # 複製影像 (避免改到原圖)
    img = img.astype(np.float32).copy()
    h, w = img.shape

    # Floyd–Steinberg 誤差擴散
    for y in range(h):
        for x in range(w):

            # 二值化 (把目前像素更新成二值化結果)
            old_pixel = img[y, x]
            new_pixel = 255 if old_pixel > 128 else 0
            img[y, x] = new_pixel

            # 計算誤差
            error = old_pixel - new_pixel

            #這裡就是「把誤差分散到鄰居像素」：
            # (整個 kernel 加起來是 16/16 = 1，代表誤差完全分散，不會丟失。)
            if x+1 < w:                      # 右邊
                img[y, x+1] += error * 7/16
            if y+1 < h and x > 0:            # 左下
                img[y+1, x-1] += error * 3/16
            if y+1 < h:                      # 正下方
                img[y+1, x]   += error * 5/16
            if y+1 < h and x+1 < w:          # 右下
                img[y+1, x+1] += error * 1/16
    # 把所有像素限制在合法範圍 0~255 
    Error_Diffusion_img = np.clip(img, 0, 255).astype(np.uint8)
    return Error_Diffusion_img
# DBS
# iterations=1 預設只做一輪搜尋
def DBS(img, iterations=1, block_size=2):
    img = img.astype(np.float32).copy()
    h, w = img.shape

    # 初始二值影像
    Binary_image = (img > 128).astype(np.float32) * 255

    coords = [(y, x) for y in range(h - block_size + 1) 
                        for x in range(w - block_size + 1)]

    for it in range(iterations):
        np.random.shuffle(coords)  # 打亂掃描順序
        for (y, x) in coords:
            # 取出 block
            block = Binary_image[y:y+block_size, x:x+block_size]
            original_error = np.sum((img[y:y+block_size, x:x+block_size] - block) ** 2)

            # 嘗試翻轉整個區塊
            flipped_block = 255 - block
            flipped_error = np.sum((img[y:y+block_size, x:x+block_size] - flipped_block) ** 2)

            # 如果翻轉後誤差更小 → 接受
            if flipped_error < original_error:
                Binary_image[y:y+block_size, x:x+block_size] = flipped_block

    return Binary_image.astype(np.uint8)

if __name__ == '__main__':

    img = cv.imread(sys.argv[1])

    n = 2
    bayer_matrix = generate_bayer_matrix(n)
    thresholds_matrix = generate_thresholds_matrix(bayer_matrix)
    # TODO:Show your picture

