import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
import heapq
import os
import cv2
from datetime import datetime
import torch

class RealsenseAPI:
    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps
        self.align = rs.align(rs.stream.color)
        self.rs = rs
        
        # 识别设备
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info(1)))

        # 启动数据流
        print(f"正在连接 RealSense 相机 ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.pipes.append(pipe)
            self.profiles[device_id] = pipe.start(config)
            print(f"已连接到相机 {i+1} ({device_id}).")

        try:
            # 获取深度传感器
            depth_sensor = self.profiles[device_id].get_device().first_depth_sensor()

            # 1. 确保红外发射器总是开启
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
            
            # 2. 提升激光功率
            if depth_sensor.supports(rs.option.laser_power):
                laser_power_range = depth_sensor.get_option_range(rs.option.laser_power)
                max_power = laser_power_range.max
                depth_sensor.set_option(rs.option.laser_power, max_power)
            else:
                print("  - 此设备不支持调整激光功率。")

            # 3. 禁用自动曝光并手动设置曝光时间
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                exposure_time_us = 8500.0
                depth_sensor.set_option(rs.option.exposure, exposure_time_us)
            else:
                print("  - 此设备不支持禁用自动曝光。")
        
        except Exception as e:
            print(f"  - 警告：设置红外投影器失败: {e}")

        
        # 初始化滤波器链
        self._initialize_filters()
        
        # 相机预热
        for _ in range(warm_start):
            self._get_frames()
        print("初始化完成。")

    def _initialize_filters(self):
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 1.0)
        self.post_filter_width = self.width // 1
        self.post_filter_height = self.height // 1

        self.threshold_filter = rs.threshold_filter(0.0, 6.0)
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial_filter = rs.spatial_filter(0.5, 20, 2, 0)
        self.temporal_filter = rs.temporal_filter(0.4, 20, 3)
        self.hole_filling_filter = rs.hole_filling_filter(1)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.filter_chain = [
            self.decimation_filter, self.threshold_filter, self.depth_to_disparity,
            self.spatial_filter, self.temporal_filter, self.hole_filling_filter,
            self.disparity_to_depth
        ]

    def _apply_filters(self, depth_frame):
        frame = depth_frame
        for f in self.filter_chain:
            frame = f.process(frame)
        return frame

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_depth(self):
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()
        depth = np.empty([num_cams, self.post_filter_height, self.post_filter_width], dtype=np.uint16)
        for i, frameset in enumerate(framesets):
            depth_frame = frameset.get_depth_frame()
            filtered_depth_frame = self._apply_filters(depth_frame)
            depth[i, :, :] = np.asanyarray(filtered_depth_frame.get_data())
        return depth
        
    def get_rgb(self):
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()
        rgb = np.empty([num_cams, self.height, self.width, 3], dtype=np.uint8)
        for i, frameset in enumerate(framesets):
            color_frame = frameset.get_color_frame()
            rgb[i, :, :, :] = np.asanyarray(color_frame.get_data())
        return rgb

    def read_frame(self, camera_index=0):
        if camera_index >= len(self.pipes):
            raise IndexError(f"相机索引 {camera_index} 超出范围。")
        frames = self.pipes[camera_index].wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        return color_img, depth_img

if __name__ == "__main__":
    # 设置输出目录
    output_dir = 'data_collection/dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化相机
    print("正在初始化RealSense相机...")
    cams = RealsenseAPI() 
    print(f"相机数量: {cams.get_num_cameras()}")
    
    image_counter = 0
    
    window_name = "RealSense Live View (Press 'A' to save, 'Q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            # 获取RGB图和处理后的深度图 
            rgb = cams.get_rgb()
            depth = cams.get_depth() 
            
            # 将RGB转为BGR用于OpenCV显示
            bgr_image = cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR)
            
            # 为小尺寸的深度图创建可视化颜色图
            depth_colormap_small = cv2.applyColorMap(
                cv2.convertScaleAbs(depth[0], alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 使用INTER_NEAREST避免在深度图中产生不存在的中间值
            depth_colormap_resized = cv2.resize(
                depth_colormap_small,
                (bgr_image.shape[1], bgr_image.shape[0]), # (width, height)
                interpolation=cv2.INTER_NEAREST
            )
            
            # 将尺寸一致的RGB图和放大后的深度图并排显示
            combined_display = np.hstack((bgr_image, depth_colormap_resized))
            
            # 在预览窗口上添加文本
            cv2.putText(combined_display, "RGB | Filtered Depth (Resized for Preview)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined_display, f"Saved: {image_counter}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, combined_display)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            
            # 保存逻辑
            if key == ord('a') or key == ord('A'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"capture_{timestamp}_{image_counter:04d}"
                
                # 保存原始尺寸的RGB图像
                rgb_path = os.path.join(output_dir, f"{base_filename}.png")
                cv2.imwrite(rgb_path, bgr_image)
                
                # 保存真实的、滤波后的、小尺寸的16位深度数据
                depth_raw_path = os.path.join(output_dir, f"{base_filename}_depth_raw.png")
                cv2.imwrite(depth_raw_path, depth[0])
                
                # # 保存与预览一致的、放大后的深度可视化图
                # depth_vis_path = os.path.join(output_dir, f"{base_filename}_depth_colormap.png")
                # cv2.imwrite(depth_vis_path, depth_colormap_resized)
                
                image_counter += 1
                print(f"已保存第 {image_counter} 组图像到: {output_dir}")
            
            # 退出逻辑
            elif key == ord('q') or key == ord('Q'):
                print("\n退出程序...")
                break
    
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        print(f"\n程序结束，共保存了 {image_counter} 组图像。")