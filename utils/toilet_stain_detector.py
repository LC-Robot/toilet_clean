import cv2
import numpy as np
import os
from utils.cv_process import segment_image


class ToiletStainDetector:
    def __init__(self, use_toilet_mask=True):
        self.use_toilet_mask = use_toilet_mask
        # 污渍颜色范围定义 (HSV色彩空间)
        self.stain_color_ranges = {
            'red_stains': [
                {'lower': np.array([0, 50, 20]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([170, 50, 20]), 'upper': np.array([180, 255, 255])}
            ],
            'brown_stains': [
                {'lower': np.array([8, 50, 20]), 'upper': np.array([20, 255, 200])}
            ],
            'black_stains': [
                {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 50])}
            ]
        }
        
        # 形态学操作核
        self.morph_kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.morph_kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.morph_kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def get_toilet_mask(self, image_path):
        """
        获取马桶区域的mask
        """
        toilet_mask = segment_image(image_path, output_mask='toilet_area_mask.png')
        if toilet_mask is not None:
            return toilet_mask
        else:
            return None

    def preprocess_image(self, image):
        """
        图像预处理
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 增强对比度 (CLAHE)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        enhanced_hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        return blurred, hsv, enhanced_hsv

    def detect_stains_by_color(self, hsv_image):
        """
        基于颜色范围检测污渍
        """
        all_stains_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        stain_info = {}
        
        for stain_type, color_ranges in self.stain_color_ranges.items():
            type_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for color_range in color_ranges:
                mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
                type_mask = cv2.bitwise_or(type_mask, mask)
            
            type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_OPEN, self.morph_kernel_small)
            type_mask = cv2.morphologyEx(type_mask, cv2.MORPH_CLOSE, self.morph_kernel_medium)
            
            stain_info[stain_type] = type_mask
            all_stains_mask = cv2.bitwise_or(all_stains_mask, type_mask)
        
        return all_stains_mask, stain_info

    def detect_stains_by_texture(self, gray_image):
        """
        基于纹理特征检测污渍
        """
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        texture_mask = cv2.bitwise_and(laplacian, adaptive_thresh)
        texture_mask = cv2.morphologyEx(texture_mask, cv2.MORPH_OPEN, self.morph_kernel_small)
        
        return texture_mask

    def detect_stains_by_brightness(self, image):
        """
        基于亮度异常检测污渍
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((15, 15), np.float32) / 225
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        brightness_diff = gray.astype(np.float32) - local_mean
        dark_threshold = -20
        dark_mask = (brightness_diff < dark_threshold).astype(np.uint8) * 255
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, self.morph_kernel_small)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, self.morph_kernel_medium)
        
        return dark_mask

    def remove_large_regions(self, mask, max_area=1000):
        """
        移除过大的区域
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) <= max_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        return filtered_mask

    def remove_small_regions(self, mask, min_area=20):
        """
        移除过小的区域
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        return filtered_mask

    def filter_by_shape_relaxed(self, mask):
        """
        更宽松的形状过滤
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                if aspect_ratio <= 12.0:
                    cv2.fillPoly(filtered_mask, [contour], 255)
        return filtered_mask

    def detect_stains(self, image_path):
        """
        主检测函数 - 综合多种方法检测污渍
        """
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                return None
        else:
            image = image_path.copy()
        
        toilet_mask = None
        if self.use_toilet_mask:
            toilet_mask = self.get_toilet_mask(image_path)
        
        blurred, hsv, enhanced_hsv = self.preprocess_image(image)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        color_mask, stain_info = self.detect_stains_by_color(enhanced_hsv)
        texture_mask = self.detect_stains_by_texture(gray)
        brightness_mask = self.detect_stains_by_brightness(blurred)
        
        if toilet_mask is not None:
            color_mask = cv2.bitwise_and(color_mask, toilet_mask)
            brightness_mask = cv2.bitwise_and(brightness_mask, toilet_mask)
            texture_mask = cv2.bitwise_and(texture_mask, toilet_mask)
            for stain_type in stain_info:
                stain_info[stain_type] = cv2.bitwise_and(stain_info[stain_type], toilet_mask)
        
        color_area = cv2.countNonZero(color_mask)
        
        if color_area > 500:
            primary_mask = color_mask.copy()
            brightness_supplement = cv2.bitwise_and(brightness_mask, cv2.bitwise_not(color_mask))
            if cv2.countNonZero(brightness_supplement) > 100:
                primary_mask = cv2.bitwise_or(primary_mask, brightness_supplement)
            final_mask = primary_mask
        else:
            final_mask = cv2.bitwise_or(color_mask, brightness_mask)
        
        final_mask = self.remove_small_regions(final_mask, min_area=50)
        final_mask = self.remove_large_regions(final_mask, max_area=5000)
        final_mask = self.filter_by_shape_relaxed(final_mask)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, self.morph_kernel_small)
        
        stain_count = len(cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        total_stain_area = cv2.countNonZero(final_mask)
        
        return {
            'original_image': image,
            'final_mask': final_mask,
            'color_mask': color_mask,
            'texture_mask': texture_mask,
            'brightness_mask': brightness_mask,
            'toilet_mask': toilet_mask,
            'stain_info': stain_info,
            'stain_count': stain_count,
            'total_area': total_stain_area
        }

    def create_visualization(self, detection_result, output_path):
        """
        创建可视化结果
        """
        if detection_result is None:
            return
        
        image = detection_result['original_image']
        final_mask = detection_result['final_mask']
        toilet_mask = detection_result['toilet_mask']
        
        result = image.copy()
        
        if toilet_mask is not None:
            toilet_contours, _ = cv2.findContours(toilet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, toilet_contours, -1, (255, 255, 0), 2)
        
        overlay = image.copy()
        overlay[final_mask > 0] = [0, 0, 255]
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        info_text = f"Stains: {detection_result['stain_count']}, Area: {detection_result['total_area']}px"
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, result)

    def create_detailed_visualization(self, detection_result, output_dir):
        """
        创建详细的分析可视化
        """
        if detection_result is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, 'color_mask.png'), detection_result['color_mask'])
        cv2.imwrite(os.path.join(output_dir, 'texture_mask.png'), detection_result['texture_mask'])
        cv2.imwrite(os.path.join(output_dir, 'brightness_mask.png'), detection_result['brightness_mask'])
        cv2.imwrite(os.path.join(output_dir, 'final_stain_mask.png'), detection_result['final_mask'])
        
        if detection_result['toilet_mask'] is not None:
            cv2.imwrite(os.path.join(output_dir, 'toilet_mask.png'), detection_result['toilet_mask'])
        
        for stain_type, mask in detection_result['stain_info'].items():
            cv2.imwrite(os.path.join(output_dir, f'{stain_type}_mask.png'), mask)