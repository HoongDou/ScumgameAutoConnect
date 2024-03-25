import cv2
import numpy as np
import pyautogui
import time
import keyboard
import sys
from threading import Thread

# 设定图片路径
image_path_1 = 'D:\\ProtableSoft\\env\\1.png'
image_path_2 = 'D:\\ProtableSoft\\env\\2.png'

# 若找不到图片则移动至此坐标
fallback_position = (960, 540)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 使用默认参数的BF匹配器
bf = cv2.BFMatcher()

def find_image(image_path, threshold=0.9, min_match_count=100):
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    template = cv2.imread(image_path, 0)

    print(f"Looking for image: {image_path}")

    if template is None:
        print(f"Failed to read the template image. Please check the file path.")
        return None
    
    # 使用SIFT检测特征点和计算描述符
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(screenshot, None)
    
    # 匹配描述符
    matches = bf.knnMatch(des1, des2, k=2)

    # 应用比率测试以找到好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    
    print(f"Number of good matches for {image_path}: {len(good_matches)}")

    if len(good_matches) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        center_x = (dst[0][0][0] + dst[2][0][0]) / 2
        center_y = (dst[0][0][1] + dst[2][0][1]) / 2
        return int(center_x), int(center_y)
    else:
        print(f"Not enough matches found for {image_path}.")
        return None

def main_loop():
    running = False
    while True:
        if keyboard.is_pressed('f8') and not running:
            running = True
            print("Start.")
            time.sleep(1.0)  # 防抖动
        elif keyboard.is_pressed('f9') and running:
            running = False
            print("Pause.")
            time.sleep(1.0)  # 防抖动
        elif keyboard.is_pressed('f10') and running:
            running = False
            print("Terminating the program.")
            sys.exit(0)

        if running:
            first_image_position = find_image(image_path_1)
            if first_image_position:
                print("First image found, searching for second image.")
                second_image_position = find_image(image_path_2)
                if second_image_position:
                    pyautogui.moveTo(*second_image_position)
                    pyautogui.click()
                    print(f"Clicked on second image at: {second_image_position}")
                else:
                    print("Second image not found, moving to fallback position.")
                    pyautogui.moveTo(*fallback_position)
                    pyautogui.click()
            else:
                print("First image not found, retrying after 15 seconds.")
                time.sleep(15)

        time.sleep(0.1)  # 以减少CPU使用率

if __name__ == '__main__':
    main_loop()
