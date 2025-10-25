import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.realsense_d435 import RealsenseAPI

def main():
    print("正在加载YOLOv8模型...")
    model_path = '/home/le/clean_ws/yolo/button_det_data/runs/detect/button_det_v2/weights/best.pt'
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"错误：无法加载模型 '{model_path}'.\n详细错误: {e}")
        return

    print("正在初始化RealSense相机...")
    try:
        cams = RealsenseAPI()
        if cams.get_num_cameras() == 0:
            print("错误：未找到RealSense相机，请检查连接。")
            return
        
        # 获取内参元组后立即解包
        cam_intrinsics = cams.get_intrinsics(camera_index=0)
        if not cam_intrinsics:
            print("错误：无法获取相机内参，程序无法继续。")
            return
        # 将元组的值赋给独立的、有意义的变量名
        fx, fy, ppx, ppy = cam_intrinsics
        print(f"相机内参已加载: fx={fx:.2f}, fy={fy:.2f}, ppx={ppx:.2f}, ppy={ppy:.2f}")
        
    except Exception as e:
        print(f"错误：初始化相机失败。\n详细错误: {e}")
        return

    print("\n相机和模型初始化完成。按 'Q' 键退出。")
    
    window_name = "YOLOv8 Button Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            rgb_image, depth_image = cams.read_frame(camera_index=0)
            if rgb_image is None or depth_image is None:
                continue 

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            results = model(bgr_image, verbose=False)

            if results[0].boxes:
                confidences = results[0].boxes.conf.cpu().numpy()
                
                if len(confidences) == 0: continue 

                best_index = np.argmax(confidences)
                
                box = results[0].boxes.xyxy.cpu().numpy().astype(int)[best_index]
                confidence = confidences[best_index]
                class_id = results[0].boxes.cls.cpu().numpy().astype(int)[best_index]
                label = model.names[class_id]

                x1, y1, x2, y2 = box
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                
                depth_value_mm = depth_image[v, u]
                
                X, Y, Z = 0.0, 0.0, 0.0
                if depth_value_mm > 0: 
                    depth_in_meters = depth_value_mm / 1000.0
                    
                    X = (u - ppx) * depth_in_meters / fx
                    Y = (v - ppy) * depth_in_meters / fy
                    Z = depth_in_meters
                
                print(f"目标: {label} | 置信度: {confidence:.2f} | 像素点: ({u}, {v}) | "
                      f"相机坐标 (X,Y,Z) in meters: ({X:.3f}, {Y:.3f}, {Z:.3f})", end='\r')

                cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(bgr_image, (u, v), 5, (0, 0, 255), -1)
                
                display_text = f"{label} Z:{Z:.2f}m"
                (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(bgr_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                cv2.putText(bgr_image, display_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow(window_name, bgr_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("\n程序已退出。")

if __name__ == "__main__":
    main()