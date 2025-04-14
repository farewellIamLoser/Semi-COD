import pyautogui
import keyboard
import time

# 全局变量，用于控制鼠标点击循环
clicking = True
click_interval = 30  # 设置点击间隔为0.5秒（可以根据需要调整）

def click_mouse():
    global clicking
    while clicking:
        pyautogui.click()
        time.sleep(click_interval)  # 设置点击间隔
        # 检查空格键是否被按下
        if keyboard.is_pressed('space'):
            clicking = False

# 执行鼠标点击
click_mouse()

print("鼠标点击已终止")
