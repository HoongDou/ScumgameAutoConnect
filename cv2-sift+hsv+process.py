import cv2
import numpy as np
import pyautogui
import time
import keyboard
import sys
import subprocess
import os
import psutil

# 设定图片路径
image_path_1 = 'D:\\1.png'
image_path_2 = 'D:\\2.png'

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 使用默认参数的BF匹配器
bf = cv2.BFMatcher()

# 定义检查进程函数
def check_process(name):
    for proc in psutil.process_iter(['pid', 'name']):
        if name.lower() in proc.info['name'].lower():
            return True
    return False

# 定义启动SCUM的函数
def start_scum():
    subprocess.run("cmd /c start steam://rungameid/513710", shell=True)  # SCUM的游戏ID是513710

def compute_histogram(image, keypoints, radius=40, mask=None, bins=8):
    histograms = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        half_size = int(radius / 2)  # 用关键点的size可能更合适，这里用固定半径简化示例
        patch = image[max(0, y-half_size):y+half_size, max(0, x-half_size):x+half_size]
        hist = cv2.calcHist([patch], [0, 1, 2], None, [bins]*3, [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    return np.array(histograms)

def find_image(image_path, threshold=0.9, min_match_count=50):
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    template = cv2.imread(image_path, cv2.IMREAD_COLOR)

    print(f"Looking for image: {image_path}")

    if template is None or screenshot is None:
        print(f"Failed to read the template or screenshot. Please check the file path.")
        return None

    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)

    hist1 = compute_histogram(template, kp1)
    hist2 = compute_histogram(screenshot, kp2)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            if np.linalg.norm(hist1[m.queryIdx] - hist2[m.trainIdx]) < threshold:
                good.append(m)

    print(f"Number of good matches for {image_path}: {len(good)}")

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w, _ = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        center_x = (dst[0][0][0] + dst[2][0][0]) / 2
        center_y = (dst[0][0][1] + dst[2][0][1]) / 2
        return int(center_x), int(center_y)
    else:
        print(f"Not enough matches found for {image_path}.")
        return None

# 定义全局变量以便在回调函数和主循环中控制运行状态
running = False

# 定义回调函数来控制程序的流程
def toggle_running():
    global running
    running = not running
    if running:
        print("Started.")
    else:
        print("Paused.")

def terminate_program():
    print("Terminating the program.")
    sys.exit(0)
    
# 注册热键
keyboard.add_hotkey('f8', toggle_running)
keyboard.add_hotkey('ctrl+f8', terminate_program)  # 使用组合键以避免与其他功能冲突

def main_loop():
    specific_position = None  # 用于存储第一张图像匹配的中心点
    while True:
        if running:
            # 检查SCUM的进程
            if not check_process('SCUM.exe'):
                print("SCUM process not found. Starting the game.")
                start_scum()
                time.sleep(60)
                continue

            # 运行图像匹配逻辑
            first_image_position = find_image(image_path_1)
            if first_image_position:
                print(f"First image found at: {first_image_position}")
                specific_position = first_image_position  # 记录第一张图的位置
            
                # 检测第二张图片
                second_image_position = find_image(image_path_2)
                if second_image_position:
                    pyautogui.moveTo(*second_image_position)
                    pyautogui.click()
                    print(f"Clicked on second image at: {second_image_position}")
                elif specific_position:
                    pyautogui.moveTo(*specific_position)
                    pyautogui.click()
                    print(f"Clicked on specific position: {specific_position}")
            else:
                print("First image not found. Retrying after 15 seconds.")
                time.sleep(15)
                continue

        time.sleep(0.1)  # 减少CPU使用率

if __name__ == '__main__':
    main_loop()
