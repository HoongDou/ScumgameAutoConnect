# ScumgameAutoConnect
use Python to make a auto-reconnect tools for bot.

# How to Install
 
 ## Python 3.12.0
 - pip
 - opencv-python、numpy、pyautogui、pyinstaller、keyboard

## Environment
 - Windows 10 x64 ver 22H2 or Windows 11 x64 ver 23H2

# How to Use
[EN]
 1. Go to the SCUM and get the screenshot of "Continue Games" and "OK".
 2. Put the screenshots into "D:\".The screenshots name should be "1.png" or "2.png" .
 3. Start the exe with F8 to toggle start/pause ,ctrl+F8 to stop the program.
 4. Make sure you've installed SCUM on steam.
[CN]
1. 进入到SCUM然后截取“继续游戏”和弹出来的“OK”界面。
2. 将截图保存在D盘根目录下，并将“继续游戏”保存为“1.png”，“OK”保存为“2.png”。
3. 打开release中的exe，并使用F8进行开启/暂停程序，ctrl+F8是退出程序。
4. 请确认：应该通过Steam安装的SCUM，否则程序会一直出错。
5. 每1分钟检测一次游戏，若游戏进程存在时每15秒检测一次图像。
# Features
 - Use opencv with sift and hsv to adapt sub-images stretching and scaling.
 - Also tried with orb but its results were not satisfactory.
