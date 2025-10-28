import cv2
import torch
import numpy as np
from ultralytics import YOLO
# 确保您的 RealsenseAPI 模块路径正确
from utils.realsense_d435 import RealsenseAPI 
import time
import os

def main():
    output_dir = '/home/le/toilet_clean/rubbish_results/薄层纸巾极限测试隔间内垂直0_7m'
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有结果将被保存到: {output_dir}")

    print("正在加载YOLOv8模型...")
    model_path = '/home/le/toilet_clean/YOLO/weights/rubbish_v1.pt'
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
    
    window_name = "YOLOv8 Rubbish Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    output_txt_path = os.path.join(output_dir, "index.txt")

    test_index = 1
    if not os.path.exists(output_txt_path):
        with open(output_txt_path, "w") as f:
            f.write("Test_ID\tStatus\tAvg_Confidence\tInference_Time(s)\n")

    try:
        while True:
            rgb_image, depth_image = cams.read_frame(camera_index=0)
            if rgb_image is None or depth_image is None:
                continue 

            bgr_image_display = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            cv2.imshow(window_name, bgr_image_display)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):
                print(f"\n开始第 {test_index} 次测试...")
                
                detected_frames = 0
                confidences_list = []
                best_frame_info = {'confidence': 0.0, 'image': None, 'box': None, 'label': ''}
                last_processed_frame = None
                
                start_time = time.time()

                for _ in range(5):
                    rgb_frame, depth_frame = cams.read_frame(camera_index=0)
                    if rgb_frame is None:
                        continue
                    
                    bgr_frame_detect = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    last_processed_frame = bgr_frame_detect
                    
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
                    
                    with open(output_txt_path, "a") as f:
                        f.write(f"test{test_index}\t{status}\t{avg_confidence:.4f}\t\t{inference_time:.4f}\n")
                    
                    if best_frame_info['image'] is not None:
                        img_to_save = best_frame_info['image']
                        x1, y1, x2, y2 = best_frame_info['box']
                        label_text = f"{best_frame_info['label']}: {best_frame_info['confidence']:.2f}"
                        
                        cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img_to_save, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                        cv2.putText(img_to_save, label_text, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        image_save_path = os.path.join(output_dir, f"test{test_index}_success_best_confidence.jpg")
                        cv2.imwrite(image_save_path, img_to_save)
                        print(f"第 {test_index} 次测试成功！结果已记录，图片已保存至 '{image_save_path}'")

                else:
                    status = "failure"
                    with open(output_txt_path, "a") as f:
                        f.write(f"test{test_index}\t{status}\tN/A\t\t{inference_time:.4f}\n")
                    print(f"第 {test_index} 次测试失败，在5帧中仅检测到 {detected_frames} 帧。")

                    if last_processed_frame is not None:
                        image_save_path = os.path.join(output_dir, f"test{test_index}_failure_last_frame.jpg")
                        cv2.imwrite(image_save_path, last_processed_frame)
                        print(f"失败时的最后一帧图像已保存至 '{image_save_path}'")
                
                test_index += 1

            elif key == ord('q'):
                break

    except Exception as e:
        print(f"\n程序运行出错: {e}")
    
    finally:
        # 确保在退出前释放相机资源
        if 'cams' in locals() and cams is not None:
            cams.release()
        cv2.destroyAllWindows()
        print("\n程序已退出。")

if __name__ == "__main__":
    main()