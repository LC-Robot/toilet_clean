import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)


# def choose_model():
#     """Initialize SAM predictor with proper parameters"""
#     model_weight = 'sam_b.pt'
#     overrides = dict(
#         task='segment',
#         mode='predict',
#         #imgsz=1024,
#         model=model_weight,
#         conf=0.01,
#         save=False
#     )
#     return SAMPredictor(overrides=overrides)

def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_l.pt'
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def segment_image_with_timing(image_path, output_mask='mask1.png'):
    """
    Enhanced version of segment_image that returns timing information
    image_path: can be either a file path (str) or a numpy array (BGR image).
    output_mask: output mask file name.
    Returns: (mask, timing_dict)
    """
    import time
    
    timing = {}
    
    target_class = "toilet"

    # 1. YOLO模型加载时间
    print("  - 加载YOLO检测模型...")
    yolo_load_start = time.time()
    detections, vis_img, yolo_model = detect_objects(image_path, target_class, return_model=True)
    yolo_load_time = time.time() - yolo_load_start
    timing['yolo_model_load'] = yolo_load_time

    if output_mask is not None:
        cv2.imwrite('detection_visualization.jpg', vis_img)

    # 2. 准备给SAM的图像
    if isinstance(image_path, str):
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 3. SAM模型加载时间
    print("  - 加载SAM分割模型...")
    sam_load_start = time.time()
    predictor = choose_model()
    predictor.set_image(image_rgb)
    sam_load_time = time.time() - sam_load_start
    timing['sam_model_load'] = sam_load_time
    
    # 4. 纯识别分割时间
    print("  - 执行分割推理...")
    inference_start = time.time()

    # 判断是否有目标检测结果
    if detections:
        # 自动选最高置信度
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"    Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # 手动点击
        print("    No detections - click on target object")
        cv2.imshow('Select Object', vis_img)

        # 初始化全局变量
        point = []
        clicked = False
        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"    Clicked at ({x}, {y})")
                point.extend([x, y])
                clicked = True  # 标记点击完成
        cv2.setMouseCallback('Select Object', click_handler)
        print("    Waiting for user click...")
        # 循环等待点击或ESC键
        while not clicked:
            key = cv2.waitKey(10)  # 10ms延迟，减少CPU占用
            if key == 27:  # ESC键退出
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()  # 安全关闭窗口

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    inference_time = time.time() - inference_start
    timing['pure_inference'] = inference_time

    # 5. 保存 mask
    if mask is not None and output_mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
    elif mask is None:
        print("[WARNING] Could not generate mask")

    return mask, timing

# def set_classes(model, target_class):
#     """Set YOLO-World model to detect specific class"""
#     model.set_classes([target_class])

def detect_objects(image_or_path, target_class=None, return_model=False):
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    return_model: if True, returns (detections, vis_img, model) for reuse
    Returns: (list of bboxes in xyxy format, visualization image) or (list of bboxes in xyxy format, visualization image, model)
    """
    model = YOLO("yolov8m-world.pt")
    if target_class:
        model.set_classes([target_class])

    results = model.predict(image_or_path, verbose=False)

    # 检查是否有结果返回
    if not results or len(results) == 0:
        # 如果没有结果，创建一个空白图像用于可视化
        if isinstance(image_or_path, str):
            vis_img = cv2.imread(image_or_path)
        else:
            vis_img = image_or_path.copy()
        return [], vis_img

    result = results[0]
    boxes = result.boxes
    vis_img = result.plot()  # Get visualized detection results

    # Extract valid detections
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.10:  # Confidence threshold 0.10
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": result.names[int(box.cls.item())]
            })

    if return_model:
        return valid_boxes, vis_img, model
    else:
        return valid_boxes, vis_img


def keep_largest_connected_component(mask):
    """
    保留mask中最大的连通区域，去除不连通的噪声
    
    Args:
        mask: 二值mask (0或255)
    
    Returns:
        cleaned_mask: 只包含最大连通区域的mask
    """
    # 使用连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:  # 只有背景，没有前景
        return mask
    
    # stats的格式: [x, y, width, height, area]
    # 第0个标签是背景，从第1个开始是前景连通区域
    areas = stats[1:, cv2.CC_STAT_AREA]  # 获取所有前景区域的面积
    
    if len(areas) == 0:
        return mask
    
    # 找到最大的连通区域（+1是因为跳过了背景标签0）
    largest_component_label = np.argmax(areas) + 1
    largest_area = areas[np.argmax(areas)]
    
    # 创建只包含最大连通区域的mask
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_component_label] = 255
    
    # 打印统计信息
    total_components = num_labels - 1  # 减去背景
    print(f"🔍 连通性分析:")
    print(f"   - 检测到 {total_components} 个连通区域")
    print(f"   - 最大区域面积: {largest_area} 像素")
    if total_components > 1:
        removed_area = np.sum(areas) - largest_area
        print(f"   - 已移除 {total_components - 1} 个噪声区域 (总计 {removed_area} 像素)")
    
    return cleaned_mask


def process_sam_results(results):
    """Process SAM results to get mask and center point"""
    if not results or not results[0].masks:
        return None, None

    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    
    # 过滤掉不连通的噪声区域，只保留最大的连通区域
    mask = keep_largest_connected_component(mask)

    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def segment_image(image_path, output_mask='mask1.png'):
    """
    image_path: can be either a file path (str) or a numpy array (BGR image).
    output_mask: output mask file name.
    """

    target_class = "toilet"

    # 2) 初步检测 - YOLO
    detections, vis_img = detect_objects(image_path, target_class)

    # 保存检测可视化结果（仅在需要保存mask时才保存可视化）
    if output_mask is not None:
        cv2.imwrite('detection_visualization.jpg', vis_img)

    # 3) 准备给 SAM 的图像 (RGB 格式)
    if isinstance(image_path, str):
        # 如果是字符串，说明是图像路径
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        # 否则假设 image_path 就是一个 BGR 的 numpy 数组
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 4) 初始化 SAM predictor
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # 5) 判断是否有目标检测结果
    if detections:
        # 自动选最高置信度
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # 手动点击
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)

        # 初始化全局变量
        point = []
        clicked = False
        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at ({x}, {y})")
                point.extend([x, y])
                clicked = True  # 标记点击完成
        cv2.setMouseCallback('Select Object', click_handler)
        print("Waiting for user click...")
        # 循环等待点击或ESC键
        while not clicked:
            key = cv2.waitKey(10)  # 10ms延迟，减少CPU占用
            if key == 27:  # ESC键退出
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()  # 安全关闭窗口

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # 6) 保存 mask
    if mask is not None and output_mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        # print(f"Segmentation saved to {output_mask}")
    elif mask is None:
        print("[WARNING] Could not generate mask")

    return mask


if __name__ == '__main__':
    seg_mask = segment_image('/home/le/clean_ws/realsense_output/capture_20251010_105701_0000.png')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
