---
title: 基于 AirSim 仿真平台与 YOLOv8s 的无人机目标检测仿真样例
date: 2025-05-05 00:10:23
tags:
    - 深度学习
    - 仿真
    - CV
categories:
    - CV
description: |
    ✈️ 最近在 AirSim 上写的一个小玩意
---

> 经过一段时间的筹备，Coldrain 花了 **1.2k** 把电脑的内存从 16G 扩充到了 **64G**，于是终于有了在电脑上同时进行实时渲染调试和模型运算的信心 💪（这个月要开始吃土了 🪨）
>
> 于是趁着五一假期，Coldrain 写了一个小程序，实现了在 **AirSim** 插件中使用键盘控制无人机自由飞行，同时利用 **YOLOv8s** 模型实现无人机的实时目标检测
>
> PS：你问为什么不用厉害点的模型？买不起 **Jetson Nano** 这种档次的设备，就算仿真有效，硬件条件也不支持迁移啊 😭（而且目前 Coldrain 也没有正式学过**边缘计算**）

## 1. 实机演示
---
先放实机演示：

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=114450778562285&bvid=BV1BSVNzhEy9&cid=25795433892&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" width="1000" height="480"></iframe>


（因为第一次给 blog 嵌入视频，折腾了一晚上，这里先把代码贴上，具体解释等到第二天再写 💦）


> 🔗 项目地址：https://github.com/ziheng5/Sharp-Eyes-target-tracking-algorithm
>
> **注**：项目持续更新中，后续会不断加入新功能

## 2. 代码详解   
---
### 2.1 导入必要的包
```python
from matplotlib.pyplot import annotate
from ultralytics import YOLO
from multiprocessing import Process
import numpy as np
import sys
import pygame
import airsim
import time
import cv2
```

- `matplotlib` 用于处理无人机摄像头获取到的图像
- `ultralytics` 用于调用 YOLO 模型
- `multiprocessing` 用于处理多进程任务
- `numpy` 用于数值计算
- `sys` 用于进程控制
- `pygame` 用于设计 AirSim 中无人机的键盘控制
- `airsim` 用于调用 AirSim 的 API
- `time` 用来控制无人机摄像头帧率
- `cv2` 用于处理无人机摄像头获取到的图像

### 2.2 设计程序框架
```python
def keyboard_control():
    # 用于实现键盘控制
    vehicle_name = "Drone1" # 这里注意切换为自己的无人机名称
    AirSim_client = airsim.MultirotorClient()
    AirSim_client.confirmConnection()
    AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
    AirSim_client.armDisarm(True, vehicle_name=vehicle_name)

def yolo_cv():
    # 用于实现调用 YOLO 模型进行实时目标检测
    vehicle_name = "Drone1" # 这里注意切换为自己的无人机名称
    AirSim_client = airsim.MultirotorClient()
    AirSim_client.confirmConnection()
    AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
    AirSim_client.armDisarm(True, vehicle_name=vehicle_name)

if __name__ == "__main__":
    # 多进程控制，避免两个任务互相干扰
    process = [Process(target=keyboard_control, args=()),
               Process(target=yolo_cv, args=())]

    [p.start() for p in process]
    [p.join() for p in process]
```

- 在 `if __name__ == "__main__":` 后面，通过 `multiprocessing` 中的 `Process` 模块，实现不同的功能函数多进程同时进行，从而提高运行效率。
- 每一个功能函数中(`keyboard_control()` 和 `yolo_cv()`) 开头都需要建立独立的 `client`，因为使用多线程的时候，无法在 `if __name__ == "__main__":` 下建立一个 `client` 并同时共享给多个进程使用。

### 2.3 无人机键盘控制设计
```python
def keyboard_control():
    # pygame settings
    pygame.init()   # pygame 初始化
    screen = pygame.display.set_mode((400, 300))    # 设置控制窗口大小
    pygame.display.set_caption("Keyboard Control")  # 设置控制窗口名称
    screen.fill((0, 0, 0))  # 用黑色 (0, 0, 0) 来填充控制窗口的背景

    # airsim settings
    ## 这里改为你要控制的无人机名称
    vehicle_name = "Drone1"
    AirSim_client = airsim.MultirotorClient()
    AirSim_client.confirmConnection()
    AirSim_client.enableApiControl(True, vehicle_name=vehicle_name)
    AirSim_client.armDisarm(True, vehicle_name=vehicle_name)
    AirSim_client.takeoffAsync(vehicle_name=vehicle_name).join()

    ## 基础的控制速度(m/s)
    vehicle_velocity = 2.0
    ## 设置临时加速度比例
    speedup_ratio = 10.0
    ## 用来设置临时加速
    speedup_flag = False
    ## 基础的偏航速率
    vehicle_yaw_rate = 5.0

    while True:
        # 主循环
        yaw_rate = 0.0
        velocity_x = 0.0
        velocity_y = 0.0
        velocity_z = 0.0

        time.sleep(0.02)    # 间隔 0.02 秒，用于减轻 CPU 负担

        # 如果关闭窗口，则进程结束
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        scan_wrapper = pygame.key.get_pressed()

        # 按下空格键加速 10 倍
        if scan_wrapper[pygame.K_SPACE]:
            scale_ratio = speedup_ratio
        else:
            scale_ratio = 1.0

        # 根据 "A" 和 "D" 按键来设置偏航速率变量
        if scan_wrapper[pygame.K_a] or scan_wrapper[pygame.K_d]:
            yaw_rate = (scan_wrapper[pygame.K_d] - scan_wrapper[pygame.K_a]) * scale_ratio * vehicle_yaw_rate

        # 根据 "UP" 和 "DOWN" 按键来设置 pitch 轴速度变量（NED 坐标系，x 为机头向前）
        if scan_wrapper[pygame.K_UP] or scan_wrapper[pygame.K_DOWN]:
            velocity_x = (scan_wrapper[pygame.K_UP] - scan_wrapper[pygame.K_DOWN]) * scale_ratio

        # 根据 "LEFT" 和 "RIGHT" 按键来设置 roll 轴速度变量（NED 坐标系，y 为正右方）
        if scan_wrapper[pygame.K_LEFT] or scan_wrapper[pygame.K_RIGHT]:
            velocity_y = (scan_wrapper[pygame.K_RIGHT] - scan_wrapper[pygame.K_LEFT]) * scale_ratio

        # 根据 "W" 和 "S" 按键来设置 z 轴速度变量（NED 坐标系，z 轴向上为负）
        if scan_wrapper[pygame.K_w] or scan_wrapper[pygame.K_s]:
            velocity_z = (scan_wrapper[pygame.K_s] - scan_wrapper[pygame.K_w]) * scale_ratio

        # 设置速度控制以及设置偏航控制
        AirSim_client.moveByVelocityBodyFrameAsync(vx=velocity_x, vy=velocity_y, vz=velocity_z, duration=0.02,
                                                   yaw_mode=airsim.YawMode(True, yaw_or_rate=yaw_rate), vehicle_name=vehicle_name)

        # press "Esc" to quit
        if scan_wrapper[pygame.K_ESCAPE]:
            AirSim_client.enableApiControl(False, vehicle_name=vehicle_name)
            AirSim_client.armDisarm(False, vehicle_name=vehicle_name)
            pygame.quit()
            sys.exit()
```

### 2.4 目标实时检测设计
```python
def yolo_cv():
    # 初始化 YOLOv8 模型
    model = YOLO('yolov8s.pt')

    # 初始化 client
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # 设置图像类型（可以是Scene, Depth, Segmentation等）
    image_type = airsim.ImageType.Scene

    # 设置摄像头名称
    camera_name = "front_center"

    # 设置显示窗口
    cv2.namedWindow("Drone FPV View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone FPV View", 1000, 600)

    # FPS 计算变量
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # 获取前端摄像头图像
            responses = client.simGetImages([
                # 不返回浮点数，不压缩
                airsim.ImageRequest(camera_name=camera_name, image_type=image_type,
                                    pixels_as_float=False, compress=False)
            ])

            response = responses[0]

            # 将图像数据转为 numpy 数组
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

            # 重塑数组为 3 通道图像
            frame = img1d.reshape(response.height, response.width, 3)

            # 使用 YOLOv8 模型来进行目标检测
            results = model.predict(frame, classes=[2])

            for result in results:
                # 绘制目标检测的 bounding box
                annotated_frame = result.plot()

                # 计算并显示FPS
                frame_count += 1
                if frame_count >= 30:  # 每30帧计算一次FPS
                    fps = frame_count / (time.time() - start_time)
                    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_count = 0
                    start_time = time.time()

                # 显示图像
                cv2.imshow("Drone FPV View", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()
```