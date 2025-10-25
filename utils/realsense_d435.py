import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
import heapq
import os
import cv2
from datetime import datetime
import torch
import json 

class RealsenseAPI:
    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps
        self.align = rs.align(rs.stream.color)
        self.rs = rs
        self.colorizer = rs.colorizer() 
        
        # 识别设备
        self.device_ls = []
        ctx = rs.context()
        for d in ctx.query_devices():
            self.device_ls.append(d.get_info(rs.camera_info.serial_number))

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
            
            profile = pipe.start(config)
            self.pipes.append(pipe)
            self.profiles[device_id] = profile
            print(f"已连接到相机 {i+1} ({device_id}).")

            try:
                device = profile.get_device()
                advnc_mode = rs.rs400_advanced_mode(device)

                json_file_path = "/home/le/perception/toilet_clean/utils/realsense-viewer.json"
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as f:
                        json_obj = json.load(f)
                    
                    # 将json对象转换为字符串以便加载
                    json_string = str(json_obj).replace("'", '\"')
                    advnc_mode.load_json(json_string)
                    print(f"  - 已为相机 {device_id} 加载 'High Accuracy' 预设。")
                else:
                    print(f"  - 警告: 未找到 'high_accuracy_preset.json' 文件，将使用默认设备设置。")

            except Exception as e:
                print(f"  - 警告: 加载高级模式配置失败: {e}")

        
        # 初始化滤波器链
        self._initialize_filters()
        
        # 相机预热
        for _ in range(warm_start):
            self._get_frames()
        print("初始化完成。")

    def _initialize_filters(self):

        # Decimation filter - 降低深度图分辨率以减少噪声
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 2)
        
        # Threshold filter - 移除超出有效距离范围的深度值
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0.1) # 10cm
        self.threshold_filter.set_option(rs.option.max_distance, 4.0) # 4m

        # Disparity transform
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # Spatial filter - 空间滤波，平滑图像并填充小空洞
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial_filter.set_option(rs.option.holes_fill, 3) 

        # Temporal filter - 时间滤波，利用历史帧进行平滑
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        # Hole filling filter - 专门的空洞填充
        self.hole_filling_filter = rs.hole_filling_filter()
        self.hole_filling_filter.set_option(rs.option.holes_fill, 1) # 1 = farest from around

        # 定义滤波器链的顺序
        self.filter_chain = [
            self.depth_to_disparity,
            self.spatial_filter,
            self.temporal_filter,
            self.disparity_to_depth,
            self.hole_filling_filter,
        ]

    def _apply_filters(self, depth_frame):
        frame = depth_frame
        for f in self.filter_chain:
            frame = f.process(frame)
        return frame.as_depth_frame() 

    def _get_frames(self):
        framesets = [pipe.wait_for_frames() for pipe in self.pipes]
        return [self.align.process(frameset) for frameset in framesets]

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_depth(self):
        framesets = self._get_frames()
        num_cams = self.get_num_cameras()
        depth = np.empty([num_cams, self.height, self.width], dtype=np.uint16)
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
            return None, None, None

        filtered_depth_frame = self._apply_filters(depth_frame)
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(filtered_depth_frame.get_data())
        return color_img, depth_img

    def read_frame_vis(self, camera_index=0):
        if camera_index >= len(self.pipes):
            raise IndexError(f"相机索引 {camera_index} 超出范围。")
        frames = self.pipes[camera_index].wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None

        filtered_depth_frame = self._apply_filters(depth_frame)
        
        depth_colormap = np.asanyarray(self.colorizer.colorize(filtered_depth_frame).get_data())

        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(filtered_depth_frame.get_data())
        return color_img, depth_img, depth_colormap

    def get_intrinsics(self, camera_index=0):
        if camera_index >= len(self.device_ls):
            print(f"错误: 相机索引 {camera_index} 超出范围。")
            return None
        
        device_id = self.device_ls[camera_index]
        profile = self.profiles[device_id]
        
        color_stream = profile.get_stream(self.rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        return (intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

    def get_valid_depth(self, depth_img, x1, y1, x2, y2, handle_length=0.04, ksize=5 ):
        h, w = depth_img.shape
        x1, x2 = max(0, x1), min(w-1, x2)
        y1, y2 = max(0, y1), min(h-1, y2)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        val_center = depth_img[cy, cx]
        if val_center > 0:
            return max(0, val_center/1000.0)

        k = ksize // 2
        patch = depth_img[max(0, cy-k):min(h, cy+k+1), max(0, cx-k):min(w, cx+k+1)]
        valid_patch = patch[patch>0]
        if valid_patch.size > 0:
            return max(0, np.median(valid_patch)/1000.0)

        corners = [
            depth_img[y1, x1], depth_img[y1, x2],
            depth_img[y2, x1], depth_img[y2, x2]
        ]
        valid_corners = [v for v in corners if v>0]
        if valid_corners:
            return max(0, np.mean(valid_corners)/1000.0 - handle_length)

        return None

    def pixels_to_camera_coords(self, pixel_list, camera_index=0):
        intrinsics = self.get_intrinsics(camera_index=camera_index)
        if intrinsics is None:
            print(f"错误: 无法获取相机 {camera_index} 的内参")
            return None
        
        fx, fy, ppx, ppy = intrinsics
        
        camera_coords = []
        for pixel in pixel_list:
            x, y, depth = pixel[0], pixel[1], pixel[2]
            
            if depth is None or depth <= 0:
                camera_coords.append([None, None, None])
                continue
            
            Z = float(depth) / 1000.0
            X = (x - ppx) * Z / fx
            Y = (y - ppy) * Z / fy
            
            camera_coords.append([X, Y, Z])
        
        return camera_coords

    def get_xyz(self, conf_thresh=0.6, yolo_dir='.', model_path='best.pt'):
        if not hasattr(self, 'model'):
            self.model = torch.hub.load(yolo_dir, "custom", path=model_path, source="local")
        self.model.conf = conf_thresh
        
        intrinsics = self.get_intrinsics(camera_index=0)
        if intrinsics is None:
            return
        fx, fy, ppx, ppy = intrinsics

        all_samples = []
        all_number = 20
        top = 5
        while True:
            color_img, depth_img, _ = self.read_frame(camera_index=0)
            if color_img is None: continue

            results = self.model(color_img)
            detections = results.pandas().xyxy[0]

            if detections is None or len(detections) != 1:
                continue

            row = detections.iloc[0]
            x1, y1 = int(row["xmin"]), int(row["ymin"])
            x2, y2 = int(row["xmax"]), int(row["ymax"])
            conf = float(row["confidence"])
            name = row["name"] if "name" in row else str(row["class"])

            depth_val = self.get_valid_depth(depth_img, x1, y1, x2, y2)
            if depth_val is None:
                continue

            Z = float(depth_val)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            X = (center_x - ppx) * Z / fx
            Y = (center_y - ppy) * Z / fy

            all_samples.append((conf, X, Y, Z, name))
            print(f"[{len(all_samples)}/{all_number}] 检测到目标: {name}, "
                  f"坐标: ({X:.3f}, {Y:.3f}, {Z:.3f}) m")

            if len(all_samples) >= all_number:
                k = min(top, len(all_samples))
                topk = heapq.nlargest(k, all_samples, key=lambda t: t[0])
                _, Xs, Ys, Zs, _ = zip(*topk)
                mean_X = float(np.mean(Xs))
                mean_Y = float(np.mean(Ys))
                mean_Z = float(np.mean(Zs))

                print(f"已收集 {all_number} 个样本，选取置信度最高的 {k} 个求均值："
                      f"({mean_X:.3f}, {mean_Y:.3f}, {mean_Z:.3f}) m")
                return (mean_X, mean_Y, mean_Z)


if __name__ == "__main__":
    output_dir = './raw_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在初始化RealSense相机...")
    cams = RealsenseAPI() 
    print(f"相机数量: {cams.get_num_cameras()}")
    
    image_counter = 0
    
    window_name = "RealSense Live View (Press 'A' to save, 'Q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            rgb, depth, depth_colormap = cams.read_frame_vis(camera_index=0)
            
            if rgb is None or depth is None:
                continue

            bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            combined_display = np.hstack((bgr_image, depth_colormap))
            
            cv2.putText(combined_display, "RGB | Filtered Depth", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined_display, f"Saved: {image_counter}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, combined_display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a') or key == ord('A'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"capture_{timestamp}_{image_counter:04d}"
                
                rgb_path = os.path.join(output_dir, f"{base_filename}.png")
                cv2.imwrite(rgb_path, bgr_image)
                
                image_counter += 1
                print(f"已保存第 {image_counter} 组图像到: {output_dir}")
            
            elif key == ord('q') or key == ord('Q'):
                print("\n退出程序...")
                break
    
    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    finally:
        cv2.destroyAllWindows()
        for pipe in cams.pipes: 
            pipe.stop()
        print(f"\n程序结束，共保存了 {image_counter} 组图像。")