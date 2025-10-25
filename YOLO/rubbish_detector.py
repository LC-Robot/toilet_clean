import cv2
import torch
import numpy as np
from ultralytics import YOLO
from utils.realsense_d435 import RealsenseAPI
import time
import os

def main():
    print("正在加载YOLOv8模型...")
    model_path = '/home/le/perception/toilet_clean/YOLO/weights/rubbish_v1.pt'
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
        
        cam_intrinsics = cams.get_intrinsics(camera_index=0)
        if not cam_intrinsics:
            print("错误：无法获取相机内参，程序无法继续。")
            return
        fx, fy, ppx, ppy = cam_intrinsics
        print(f"相机内参已加载: fx={fx:.2f}, fy={fy:.2f}, ppx={ppx:.2f}, ppy={ppy:.2f}")
        
    except Exception as e:
        print(f"错误：初始化相机失败。\n详细错误: {e}")
        return

    print("\n相机和模型初始化完成。按 'A' 键开始检测，按 'Q' 键退出。")
    
    window_name = "YOLOv8 Button Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    test_index = 1
    if not os.path.exists("index.txt"):
        with open("index.txt", "w") as f:
            f.write("Test_ID\tStatus\tAvg_Confidence\tInference_Time(s)\n")

    try:
        while True:
            rgb_image, depth_image = cams.read_frame(camera_index=0)
            if rgb_image is None or depth_image is None:
                continue 

            bgr_image_display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            # 在循环外显示图像，保持可视化
            cv2.imshow(window_name, bgr_image_display)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                print(f"\n开始第 {test_index} 次测试...")
                
                detected_frames = 0
                confidences_list = []
                best_frame_info = {'confidence': 0.0, 'image': None, 'box': None, 'label': ''}
                
                start_time = time.time()

                for _ in range(5):
                    # 获取新的一帧进行检测
                    rgb_frame, depth_frame = cams.read_frame(camera_index=0)
                    if rgb_frame is None:
                        continue
                    
                    bgr_frame_detect = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    results = model(bgr_frame_detect, verbose=False)

                    if results[0].boxes and len(results[0].boxes.conf) > 0:
                        detected_frames += 1
                        
                        confidences = results[0].boxes.conf.cpu().numpy()
                        best_index = np.argmax(confidences)
                        confidence = confidences[best_index]
                        confidences_list.append(confidence)
                        
                        if confidence > best_frame_info['confidence']:
                            box = results[0].boxes.xyxy.cpu().numpy().astype(int)[best_index]
                            class_id = results[0].boxes.cls.cpu().numpy().astype(int)[best_index]
                            label = model.names[class_id]
                            
                            best_frame_info['confidence'] = confidence
                            best_frame_info['image'] = bgr_frame_detect.copy()
                            best_frame_info['box'] = box
                            best_frame_info['label'] = label

                end_time = time.time()
                inference_time = end_time - start_time

                if detected_frames == 5:
                    avg_confidence = np.mean(confidences_list)
                    status = "success"
                    
                    # 记录到txt文件
                    with open("index.txt", "a") as f:
                        f.write(f"test{test_index}\t{status}\t{avg_confidence:.4f}\t\t{inference_time:.4f}\n")
                    
                    # 保存置信度最高的图片
                    if best_frame_info['image'] is not None:
                        img_to_save = best_frame_info['image']
                        x1, y1, x2, y2 = best_frame_info['box']
                        label_text = f"{best_frame_info['label']}: {best_frame_info['confidence']:.2f}"
                        
                        cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img_to_save, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                        cv2.putText(img_to_save, label_text, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        cv2.imwrite(f"test{test_index}_best_confidence.jpg", img_to_save)
                        print(f"第 {test_index} 次测试成功！结果已记录，最高置信度图片已保存。")

                else:
                    status = "failure"
                    # 记录到txt文件
                    with open("index.txt", "a") as f:
                        f.write(f"test{test_index}\t{status}\tN/A\t\t{inference_time:.4f}\n")
                    print(f"第 {test_index} 次测试失败，连续5帧未完全检测到目标。")
                
                test_index += 1

            elif key == ord('q'):
                break

    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    finally:
        cv2.destroyAllWindows()
        print("\n程序已退出。")

if __name__ == "__main__":
    main()