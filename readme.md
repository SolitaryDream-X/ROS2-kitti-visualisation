# ROS2实现kitti数据可视化

  本仓库代码实现了使用ROS2进行kitti的数据可视化

## 测试环境

- 系统：Garuda（宿主机）+ Ubuntu 24.04LTS（distrobox）

- ROS2版本：jazzy

- Python版本：3.12.8

![env](https://github.com/SolitaryDream-X/ROS2-kitti-visualisation/blob/main/img/env.png?raw=true)

![distrobox-env](https://github.com/SolitaryDream-X/ROS2-kitti-visualisation/blob/main/img/distrobox-env.png?raw=true)

## 项目结构

![result](https://github.com/SolitaryDream-X/ROS2-kitti-visualisation/blob/main/img/tree.png?raw=true)

## 最终效果

![result](https://github.com/SolitaryDream-X/ROS2-kitti-visualisation/blob/main/img/result.png?raw=true)

[演示视频(bilibili)](https://www.bilibili.com/video/BV1pmfJYUEbX/?share_source=copy_web&vd_source=16b9a6caf533993510f852ff67f71551)

## 运行

- 请先修改源码里的数据路径为你使用的路径

- 在`Publisher`文件夹下打开终端

- `source`你的ROS2(类似这样`source /opt/ros/jazzy/setup.zsh`)

- 执行`colcon build`

- 执行`source ./install/setup.zsh`(或者是`source ./install/setup.bash`)

- 执行`ros2 run kitti_visualisation kitti_visualisation`

- 打开rviz2,添加topic即可看到效果
