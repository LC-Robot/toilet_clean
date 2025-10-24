import cv2
import numpy as np
import os
import time
import math
from utils.cv_process import segment_image, segment_image_with_timing
from utils.depth_segmentation import segment_toilet_depth_complete
from utils.toilet_stain_detector import ToiletStainDetector
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class IntegratedToiletAnalyzer:
    def __init__(self, adaptive_ratio=0.5, depth_kernel_size=5):
        self.adaptive_ratio = adaptive_ratio
        self.depth_kernel_size = depth_kernel_size
        
        self._rgb = None
        self._depth = None
        self._total_mask = None
        self._rim_mask = None
        self._inner_mask = None
        self._drain_hole_mask = None  # 新增：排水口mask
        self._stain_results = None
        self._is_analyzed = False
        
    def analyze_toilet(self, rgb_input, depth_input, output_dir='toilet_analysis_results'):
        total_start_time = time.time()
        timing_stats = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 步骤1: 分割整个马桶区域
        step1_start = time.time()
        if isinstance(rgb_input, str):
            total_mask, segmentation_timing = segment_image_with_timing(rgb_input, output_mask=None)
        else:
            total_mask, segmentation_timing = self._segment_image_array(rgb_image=rgb_input, output_mask=None)
        
        step1_time = time.time() - step1_start
        timing_stats['step1_toilet_segmentation'] = step1_time
        timing_stats['step1_yolo_model_load'] = segmentation_timing['yolo_model_load']
        timing_stats['step1_sam_model_load'] = segmentation_timing['sam_model_load']
        timing_stats['step1_pure_inference'] = segmentation_timing['pure_inference']
        
        if total_mask is None:
            return None
        
        total_mask = self._fill_holes_in_mask(total_mask)
        cv2.imwrite(f'{output_dir}/total_mask.png', total_mask)
        
        # 步骤2: 基于深度分割inner和rim
        step2_start = time.time()
        if isinstance(rgb_input, str):
            rgb = cv2.imread(rgb_input)
            depth = cv2.imread(depth_input, cv2.IMREAD_ANYDEPTH)
        else:
            rgb = rgb_input.copy()
            depth = depth_input.copy()
        
        self._rgb = rgb
        self._depth = depth
        self._total_mask = total_mask
        
        depth_result = segment_toilet_depth_complete(
            rgb=rgb,
            depth=depth,
            mask=total_mask,
            adaptive_ratio=self.adaptive_ratio,
            kernel_size=self.depth_kernel_size,
            output_dir='toilet_segmentation_output',
            verbose=False,
            detect_drain_hole=True,  # 启用排水口检测
            drain_depth_percentile=95
        )
        
        if depth_result is None:
            return None
        
        rim_mask_final = depth_result['rim_mask']
        inner_mask_final = depth_result['inner_mask']
        drain_hole_mask = depth_result['drain_hole_mask']  # 获取排水口mask
        
        cv2.imwrite(f'{output_dir}/rim_mask.png', rim_mask_final)
        cv2.imwrite(f'{output_dir}/inner_mask.png', inner_mask_final)
        
        if drain_hole_mask is not None:
            cv2.imwrite(f'{output_dir}/drain_hole_mask.png', drain_hole_mask)
        
        self._rim_mask = rim_mask_final
        self._inner_mask = inner_mask_final
        self._drain_hole_mask = drain_hole_mask
        
        step2_time = time.time() - step2_start
        timing_stats['step2_total'] = step2_time
        
        # 步骤3: 对三个区域分别进行污渍检测
        step3_start = time.time()
        detector = ToiletStainDetector(use_toilet_mask=False)
        results = {}
        
        detect_total_start = time.time()
        total_result = self._detect_stains_in_region(
            detector, rgb, total_mask, inner_mask_final,
            region_name="Total",
            output_dir=output_dir
        )
        timing_stats['step3_detect_total'] = time.time() - detect_total_start
        results['total'] = total_result
        
        detect_rim_start = time.time()
        rim_result = self._detect_stains_in_region(
            detector, rgb, rim_mask_final, inner_mask_final,
            region_name="Rim",
            output_dir=output_dir
        )
        timing_stats['step3_detect_rim'] = time.time() - detect_rim_start
        results['rim'] = rim_result
        
        detect_inner_start = time.time()
        # 如果检测到排水口，从inner_mask中排除排水口区域
        inner_mask_for_stain_detection = inner_mask_final.copy()
        if self._drain_hole_mask is not None:
            inner_mask_for_stain_detection = cv2.subtract(inner_mask_for_stain_detection, self._drain_hole_mask)
        
        inner_result = self._detect_stains_in_region(
            detector, rgb, inner_mask_for_stain_detection, inner_mask_final,
            region_name="Inner",
            output_dir=output_dir
        )
        timing_stats['step3_detect_inner'] = time.time() - detect_inner_start
        results['inner'] = inner_result
        
        step3_time = time.time() - step3_start
        timing_stats['step3_total'] = step3_time
        
        # 步骤4: 生成综合报告
        step4_start = time.time()
        total_time = time.time() - total_start_time
        timing_stats['total_time'] = total_time
        
        self._generate_report(rgb, total_mask, rim_mask_final, inner_mask_final, 
                             results, output_dir, timing_stats)
        step4_time = time.time() - step4_start
        timing_stats['step4_generate_report'] = step4_time
        
        self._stain_results = results
        self._is_analyzed = True
        
        return results
    
    def _segment_image_array(self, rgb_image, output_mask=None):
        from utils.cv_process import segment_image
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
            cv2.imwrite(temp_path, rgb_image)
        
        try:
            mask = segment_image(temp_path, output_mask=output_mask)
            timing_stats = {
                'yolo_model_load': 0.0,
                'sam_model_load': 0.0,
                'pure_inference': 0.0
            }
            return mask, timing_stats
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _fill_holes_in_mask(self, mask):
        if mask is None or cv2.countNonZero(mask) == 0:
            return mask

        mask_floodfill = mask.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)

        holes = cv2.bitwise_not(mask_floodfill)

        filled_mask = cv2.bitwise_or(mask, holes)

        return filled_mask
    
    
    def _detect_stains_in_region(self, detector, rgb_image, region_mask, 
                                 inner_mask_for_exclusion,
                                 region_name, output_dir):
        image = rgb_image
        h, w = image.shape[:2]
        exclusion_mask_top = np.ones((h, w), dtype=np.uint8) * 255 

        contours, _ = cv2.findContours(inner_mask_for_exclusion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            x, y, w_bbox, h_bbox = cv2.boundingRect(main_contour)
            top_points = []
            for point in main_contour:
                if point[0][1] < y + h_bbox * 0.25:
                    top_points.append(point[0])
            
            if len(top_points) > 5:
                top_points = np.array(top_points, dtype=np.float32)

                line_params = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                vx = line_params[0][0]
                vy = line_params[1][0]

                angle = math.degrees(math.atan2(vy, vx))

                min_x = int(np.min(top_points[:, 0]))
                max_x = int(np.max(top_points[:, 0]))
                original_width = max_x - min_x
                LENGTH_SCALE_FACTOR = 1.5
                rect_width = original_width * LENGTH_SCALE_FACTOR                
                
                EXCLUSION_HEIGHT = 50
                rect_height = EXCLUSION_HEIGHT
                
                center_on_line = (np.mean(top_points[:, 0]), np.mean(top_points[:, 1]))
                normal_vector = (vy, -vx) 
                
                OFFSET_FROM_INNER_TOP = 60
                offset_distance = OFFSET_FROM_INNER_TOP + (rect_height / 2)
                
                center_x = center_on_line[0] + offset_distance * normal_vector[0]
                center_y = center_on_line[1] + offset_distance * normal_vector[1]

                rotated_rect = ((center_x, center_y), (rect_width, rect_height), angle)
                box_points = cv2.boxPoints(rotated_rect)
                box_points = np.int_(box_points)
                
                cv2.fillPoly(exclusion_mask_top, [box_points], 0)
                
        blurred, hsv, enhanced_hsv = detector.preprocess_image(image)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        color_mask, stain_info = detector.detect_stains_by_color(enhanced_hsv)
        texture_mask = detector.detect_stains_by_texture(gray)
        brightness_mask = detector.detect_stains_by_brightness(blurred)
        
        color_mask = cv2.bitwise_and(color_mask, region_mask)
        brightness_mask = cv2.bitwise_and(brightness_mask, region_mask)
        texture_mask = cv2.bitwise_and(texture_mask, region_mask)
        
        for stain_type in stain_info:
            stain_info[stain_type] = cv2.bitwise_and(stain_info[stain_type], region_mask)
        
        color_area = cv2.countNonZero(color_mask)
        
        if color_area > 500:
            primary_mask = color_mask.copy()
            brightness_supplement = cv2.bitwise_and(brightness_mask, cv2.bitwise_not(color_mask))
            if cv2.countNonZero(brightness_supplement) > 100:
                primary_mask = cv2.bitwise_or(primary_mask, brightness_supplement)
            final_mask = primary_mask
        else:
            final_mask = cv2.bitwise_or(color_mask, brightness_mask)
        
        final_mask = detector.remove_small_regions(final_mask, min_area=50)
        final_mask = detector.remove_large_regions(final_mask, max_area=5000)
        final_mask = detector.filter_by_shape_relaxed(final_mask)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, detector.morph_kernel_small)
        
        final_mask = cv2.bitwise_and(final_mask, exclusion_mask_top)
        
        stain_count = len(cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        total_stain_area = cv2.countNonZero(final_mask)
        region_area = cv2.countNonZero(region_mask)
        contamination_rate = (total_stain_area / region_area * 100) if region_area > 0 else 0
        
        result_vis = image.copy()
        
        region_contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_vis, region_contours, -1, (255, 255, 0), 2) 
        
        exclusion_contours, _ = cv2.findContours(cv2.bitwise_not(exclusion_mask_top), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_vis, exclusion_contours, -1, (0, 0, 255), 1) 

        overlay = image.copy()
        overlay[final_mask > 0] = [0, 0, 255]
        result_vis = cv2.addWeighted(result_vis, 0.7, overlay, 0.3, 0)
        
        stain_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_vis, stain_contours, -1, (0, 255, 0), 2)  
        
        title = f"{region_name} Region - Stains: {stain_count}, Area: {total_stain_area}px ({contamination_rate:.1f}%)"
        cv2.putText(result_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        output_path = f'{output_dir}/{region_name.lower()}_stain_detection.jpg'
        cv2.imwrite(output_path, result_vis)
        
        return {
            'region_name': region_name,
            'region_area': region_area,
            'stain_count': stain_count,
            'stain_area': total_stain_area,
            'contamination_rate': contamination_rate,
            'final_mask': final_mask,
            'visualization': result_vis
        }

    def _generate_report(self, rgb, total_mask, rim_mask, inner_mask, 
                        results, output_dir, timing_stats):
        h, w = rgb.shape[:2]
        report_img = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        overlay = rgb.copy()
        overlay[rim_mask > 0] = cv2.addWeighted(overlay[rim_mask > 0], 0.6, 
                                               np.full_like(overlay[rim_mask > 0], [255, 0, 0]), 0.4, 0)
        overlay[inner_mask > 0] = cv2.addWeighted(overlay[inner_mask > 0], 0.6, 
                                                 np.full_like(overlay[inner_mask > 0], [0, 255, 0]), 0.4, 0)
        cv2.putText(overlay, "Region Segmentation", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "Blue: Rim, Green: Inner", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        report_img[0:h, 0:w] = overlay
        
        report_img[0:h, w:w*2] = results['total']['visualization']
        report_img[h:h*2, 0:w] = results['rim']['visualization']
        report_img[h:h*2, w:w*2] = results['inner']['visualization']
        
        cv2.imwrite(f'{output_dir}/analysis_report.jpg', report_img)
        
    def get_stain_mask(self, rgb_input, depth_input, region_type='rim', include_depth_info=False):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        if region_type not in ['rim', 'inner']:
            raise ValueError("region_type 必须是 'rim' 或 'inner'")
        
        stain_result = self._stain_results.get(region_type)
        if stain_result is None:
            return None
        
        stain_mask = stain_result['final_mask']
        
        if not include_depth_info:
            return stain_mask
        
        depth_values = self._depth[stain_mask > 0]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            depth_stats = {'mean': 0, 'min': 0, 'max': 0, 'median': 0}
        else:
            depth_stats = {
                'mean': float(np.mean(valid_depths)),
                'min': float(np.min(valid_depths)),
                'max': float(np.max(valid_depths)),
                'median': float(np.median(valid_depths))
            }
        
        return {
            'stain_mask': stain_mask,
            'depth_values': valid_depths,
            'depth_stats': depth_stats
        }
    
    def is_cleaned(self, rgb_input, depth_input, region_type='total', threshold=2.0):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        
        if region_type not in ['total', 'rim', 'inner']:
            raise ValueError("region_type 必须是 'total', 'rim' 或 'inner'")
        
        stain_result = self._stain_results.get(region_type)
        if stain_result is None:
            return None
        
        # contamination_rate = stain_result['contamination_rate']
        # is_cleaned = contamination_rate < threshold

        # TODO: 根据实际情况判断是否使用污染阈值or污渍数量
        stain_count = stain_result['stain_count']
        is_cleaned = stain_count < 1
        
        return {
            'is_cleaned': is_cleaned,
            # 'contamination_rate': contamination_rate,
            'stain_count': stain_result['stain_count'],
            'stain_area': stain_result['stain_area'],
            'region_area': stain_result['region_area']
        }
    
    def detect_toilet(self, rgb_input):
        try:
            if isinstance(rgb_input, str):
                total_mask, _ = segment_image_with_timing(rgb_input, output_mask=None)
            else:
                total_mask, _ = self._segment_image_array(rgb_image=rgb_input, output_mask=None)
            
            if total_mask is None or cv2.countNonZero(total_mask) == 0:
                return {'detected': False, 'confidence': 0.0, 'toilet_area': 0}
            
            toilet_area = cv2.countNonZero(total_mask)
            
            if isinstance(rgb_input, str):
                img = cv2.imread(rgb_input)
            else:
                img = rgb_input
            
            image_area = img.shape[0] * img.shape[1]
            area_ratio = toilet_area / image_area
            
            if 0.1 <= area_ratio <= 0.8:
                confidence = 0.9
            elif 0.05 <= area_ratio < 0.1 or 0.8 < area_ratio <= 0.9:
                confidence = 0.7
            else:
                confidence = 0.5
            
            return {
                'detected': True,
                'confidence': confidence,
                'toilet_area': toilet_area
            }
            
        except Exception:
            return {'detected': False, 'confidence': 0.0, 'toilet_area': 0}
    
    def get_toilet_mask(self, rgb_input, depth_input):
        if self._is_analyzed and self._total_mask is not None:
            return self._total_mask
        
        self.analyze_toilet(rgb_input, depth_input)
        return self._total_mask
    
    def get_rim_mask(self, rgb_input, depth_input):
        if self._is_analyzed and self._rim_mask is not None:
            return self._rim_mask
        
        self.analyze_toilet(rgb_input, depth_input)
        return self._rim_mask
    
    def get_drain_hole_mask(self, rgb_input, depth_input):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        
        return self._drain_hole_mask
    
    def get_rim_bottom_left_point(self, rgb_input, depth_input):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        
        if self._rim_mask is None:
            return {'x': 0, 'y': 0, 'depth': 0, 'found': False}
        
        # 找到rim_mask中所有非零点
        y_coords, x_coords = np.where(self._rim_mask > 0)
        
        if len(y_coords) == 0:
            return {'x': 0, 'y': 0, 'depth': 0, 'found': False}
        
        scores = y_coords - x_coords
        
        best_point_index = np.argmax(scores)
        
        bottom_left_x = x_coords[best_point_index]
        bottom_left_y = y_coords[best_point_index]
        
        # 获取该点的深度值
        depth_value = int(self._depth[bottom_left_y, bottom_left_x])
        
        return {
            'x': int(bottom_left_x),
            'y': int(bottom_left_y),
            'depth': depth_value,
            'found': True
        }
    
    def get_rim_skeleton(self, rgb_input, depth_input):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        
        if self._rim_mask is None or self._inner_mask is None:
            return None
        
        # 使用形态学细化提取骨架
        skeleton = np.zeros_like(self._rim_mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = self._rim_mask.copy()
        
        while True:
            # 开运算
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = cv2.subtract(eroded, opened)
            skeleton = cv2.bitwise_or(skeleton, subset)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        h, w = skeleton.shape
        exclusion_mask_top = np.ones((h, w), dtype=np.uint8) * 255
        
        contours, _ = cv2.findContours(self._inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # 提取顶部轮廓点
            x, y, w_bbox, h_bbox = cv2.boundingRect(main_contour)
            top_points = []
            for point in main_contour:
                if point[0][1] < y + h_bbox * 0.25:  # 顶部25%的点
                    top_points.append(point[0])
            
            if len(top_points) > 5:
                top_points = np.array(top_points, dtype=np.float32)
                
                # 拟合一条直线
                line_params = cv2.fitLine(top_points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                vx = line_params[0][0]
                vy = line_params[1][0]
                x0 = line_params[2][0]
                y0 = line_params[3][0]
                
                # 设置偏移距离（向上偏移，即向环线外侧偏移）
                OFFSET_FROM_EDGE = 10  # 可调参数：偏离边缘环线的距离（像素）
                
                # 计算法向量（指向上方）
                normal_x = -vy
                normal_y = vx
                normal_length = np.sqrt(normal_x**2 + normal_y**2)
                normal_x /= normal_length
                normal_y /= normal_length
                
                # 偏移后的直线上的一点
                x0_offset = x0 + normal_x * OFFSET_FROM_EDGE
                y0_offset = y0 + normal_y * OFFSET_FROM_EDGE
                
                # 对mask的每一列，计算裁剪线的y坐标，将其上方区域设为0
                for col in range(w):
                    if abs(vx) > 0.001:  
                        y_on_line = y0_offset + vy / vx * (col - x0_offset)
                        
                        # 将此y值以上的区域设为0
                        if 0 <= y_on_line < h:
                            exclusion_mask_top[:int(y_on_line), col] = 0
                    else:
                        if col < x0_offset:
                            exclusion_mask_top[:, col] = 0
        
        # 应用排除mask到骨架
        skeleton = cv2.bitwise_and(skeleton, exclusion_mask_top)
        
        # 使用连通组件分析找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            skeleton, connectivity=8
        )
        
        if num_labels > 1: 
            areas = stats[1:, cv2.CC_STAT_AREA]  
            
            if len(areas) > 0:
                # 找到最大的连通组件
                largest_label = np.argmax(areas) + 1  # 
                
                # 创建只包含最大组件的mask
                skeleton_cleaned = np.zeros_like(skeleton)
                skeleton_cleaned[labels == largest_label] = 255
                
                total_points = np.sum(areas)
                skeleton = np.zeros_like(skeleton)
                
                for i, area in enumerate(areas):
                    if area > total_points * 0.008:  
                        skeleton[labels == (i + 1)] = 255
                
                if cv2.countNonZero(skeleton) == 0:
                    skeleton = skeleton_cleaned
        
        # 获取清理后骨架上所有点的坐标
        y_coords, x_coords = np.where(skeleton > 0)
        
        if len(y_coords) == 0:
            return {
                'skeleton_mask': skeleton,
                'points': [],
                'depths': [],
                'depth_stats': {'mean': 0, 'min': 0, 'max': 0, 'median': 0},
                'skeleton_length': 0
            }
        
        # 构建点列表
        points = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
        order_point = self.filter_and_sort_coordinates(points)
        bspline_points = self.fit_bspline_to_points(order_point)
        
        # 获取每个点的深度值
        depths = [int(self._depth[y, x]) for x, y in zip(x_coords, y_coords)]
        
        # 过滤掉深度为0的无效点
        valid_mask = np.array(depths) > 0
        valid_depths = np.array(depths)[valid_mask]
        
        # 计算深度统计
        if len(valid_depths) > 0:
            depth_stats = {
                'mean': float(np.mean(valid_depths)),
                'min': float(np.min(valid_depths)),
                'max': float(np.max(valid_depths)),
                'median': float(np.median(valid_depths))
            }
        else:
            depth_stats = {'mean': 0, 'min': 0, 'max': 0, 'median': 0}
            
        return {
            'skeleton_mask': skeleton,
            'points': bspline_points,#排序后的点
            'depths': depths,
            'depth_stats': depth_stats,
            'skeleton_length': len(order_point)
        }
    
    def get_stain_deepest_points(self, rgb_input, depth_input, region_type='inner'):
        if not self._is_analyzed:
            self.analyze_toilet(rgb_input, depth_input)
        
        if region_type not in ['rim', 'inner']:
            raise ValueError("region_type 必须是 'rim' 或 'inner'")
        
        stain_mask = self.get_stain_mask(rgb_input, depth_input, region_type=region_type, include_depth_info=False)
        
        if stain_mask is None or cv2.countNonZero(stain_mask) == 0:
            return []
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            stain_mask, connectivity=8
        )
        
        stains_info = []
        
        for stain_id in range(1, num_labels):
            current_stain_mask = (labels == stain_id).astype(np.uint8) * 255
            
            y_coords, x_coords = np.where(current_stain_mask > 0)
            
            if len(y_coords) == 0:
                continue
            
            depth_values = self._depth[y_coords, x_coords]
            
            # 过滤掉无效深度（<=0）
            valid_mask = depth_values > 0
            
            if np.sum(valid_mask) == 0:
                deepest_point = {
                    'x': int(x_coords[0]),
                    'y': int(y_coords[0]),
                    'depth': 0
                }
            else:
                valid_y = y_coords[valid_mask]
                valid_x = x_coords[valid_mask]
                valid_depths = depth_values[valid_mask]
                
                max_depth_idx = np.argmax(valid_depths)
                deepest_point = {
                    'x': int(valid_x[max_depth_idx]),
                    'y': int(valid_y[max_depth_idx]),
                    'depth': int(valid_depths[max_depth_idx])
                }
            
            # === 计算中心点 ===
            center_x = int(round(centroids[stain_id][0]))
            center_y = int(round(centroids[stain_id][1]))
            
            h, w = self._depth.shape
            center_x = max(0, min(w - 1, center_x))
            center_y = max(0, min(h - 1, center_y))
            
            center_depth = int(self._depth[center_y, center_x])
            
            if center_depth <= 0:
                # 在3x3邻域内搜索
                search_radius = 5
                found_valid = False
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        ny = center_y + dy
                        nx = center_x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if current_stain_mask[ny, nx] > 0:
                                neighbor_depth = int(self._depth[ny, nx])
                                if neighbor_depth > 0:
                                    center_depth = neighbor_depth
                                    found_valid = True
                                    break
                    if found_valid:
                        break
            
            center_point = {
                'x': center_x,
                'y': center_y,
                'depth': center_depth
            }
            
            # 添加到结果列表
            stains_info.append({
                'stain_id': stain_id,
                'deepest_point': deepest_point,
                'center_point': center_point,
                'stain_area': stats[stain_id, cv2.CC_STAT_AREA]
            })
        
        return stains_info
    
    def fit_bspline_to_points(self,points: list[tuple[float, float]], s: float = 500.0, k: int = 3, num_points: int = 100) -> list[tuple[float, float]]:
        """
        根据给定的二维坐标点列表，拟合B样条曲线并返回平滑曲线上的点。

        参数:
        points (List[Tuple[float, float]]): 二维坐标点的列表，格式为 [(x1, y1), (x2, y2), ...]。
        s (float): 平滑因子。s=0表示曲线必须穿过所有点（插值）。值越大，曲线越平滑。
        k (int): B样条的阶数（degree）。默认为3，即三次B样条。
        num_points (int): 用于生成平滑曲线的采样点数量。

        返回:
        List[Tuple[float, float]]: 一个包含拟合后平滑曲线上 (x, y) 坐标的列表。
                                如果输入无效，则返回一个空列表。
        """
        # --- 1. 输入验证 ---
        # 检查输入是否为列表或Numpy数组，并且点的数量是否足够进行k阶拟合
        if not isinstance(points, (list, np.ndarray)) or len(points) < k + 1:
            print(f"错误：输入数据必须是列表或numpy数组，并且至少包含 {k + 1} 个点才能进行 {k} 阶拟合。")
            return []

        # --- 2. 数据准备 ---
        try:
            # 将输入点转换为Numpy数组，并分离x和y坐标
            points_array = np.array(points)
            x = points_array[:, 0]
            y = points_array[:, 1]
        except (IndexError, TypeError):
            print("错误：请确保 'points' 是一个有效的二维坐标列表，例如 [(x1, y1), ...]")
            return []

        # --- 3. B样条曲线拟合 ---
        # splprep (Spline Preparation) 返回tck（包含节点、系数和阶数的元组）和u（参数值）
        # s是平滑度参数，k是样条阶数
        tck, u = splprep([x, y], s=s, k=k)

        # --- 4. 生成平滑曲线上的点 ---
        # 创建一个新的、更密集的参数化变量u_new，用于生成平滑曲线
        u_new = np.linspace(u.min(), u.max(), num_points)
        
        # splev (Spline Evaluation) 在新的参数点上评估样条曲线，得到新的x和y坐标
        x_new, y_new = splev(u_new, tck)

        # --- 5. 格式化并返回结果 ---
        # 将新的x和y坐标组合成一个元组列表
        x_int = x_new.astype(int)
        y_int = y_new.astype(int)
        fitted_points = list(zip(x_int, y_int))
        
        return fitted_points

    def filter_and_sort_coordinates(self,coordinates: list[tuple[int, int]], 
        distance_threshold: int = 10 ) -> list[tuple[int, int]]:
        """
        根据特定规则筛选和排序坐标点。

        筛选规则:
        1. 针对每一个唯一的 y 值，找出其对应的 x 坐标的最小值和最大值。
        2. 如果某个 y 值对应的 max_x 和 min_x 之差小于 `distance_threshold`，
        那么所有与该 y 值相关的点都将被丢弃。此规则也自动处理了只有一个 x 值的情况。

        排序规则:
        1. 从最小的 y 值开始遍历到最大的 y 值，依次添加 (min_x, y) 点。
        2. 接着从最大的 y 值反向遍历到最小的 y 值，依次添加 (max_x, y) 点。

        Args:
            coordinates (List[Tuple[int, int]]): 包含 (x, y) 坐标元组的列表。
            distance_threshold (int, optional): x 坐标之间的最小距离阈值。
                                                如果 max_x - min_x < threshold，这些点将被丢弃。
                                                默认为 10。

        Returns:
            List[Tuple[int, int]]: 一个经过筛选和排序的、包含新坐标点顺序的列表。
        """
        if not coordinates:
            print("输入的坐标列表为空。")
            return []

        # --- 步骤 1: 为每个 y 值找到 x 坐标的范围 ---
        y_to_x_extremes = {}
        for x, y in coordinates:
            if y not in y_to_x_extremes:
                y_to_x_extremes[y] = [x, x]  # [min_x, max_x]
            else:
                y_to_x_extremes[y][0] = min(x, y_to_x_extremes[y][0])
                y_to_x_extremes[y][1] = max(x, y_to_x_extremes[y][1])

        # --- 步骤 2: 应用距离阈值进行最终筛选 ---
        filtered_y_to_x_extremes = {}
        for y, (min_x, max_x) in y_to_x_extremes.items():
            # 核心条件：只有当 x 跨度足够大时，才保留该 y 层级
            if (max_x - min_x) >= distance_threshold:
                filtered_y_to_x_extremes[y] = [min_x, max_x]
        
        if not filtered_y_to_x_extremes:
            print("筛选后没有剩下任何有效的坐标数据。")
            return []

        # --- 步骤 3: 基于筛选后的数据执行双向排序 ---
        ordered_points = []
        
        # 按 y 值升序排序
        sorted_y_keys_asc = sorted(filtered_y_to_x_extremes.keys())
        
        # 正向遍历：从最小的 y 到最大的 y，添加 min_x
        for y in sorted_y_keys_asc:
            min_x = filtered_y_to_x_extremes[y][0]
            ordered_points.append((min_x, y))

        # 反向遍历：从最大的 y 到最小的 y，添加 max_x
        for y in reversed(sorted_y_keys_asc):
            max_x = filtered_y_to_x_extremes[y][1]
            ordered_points.append((max_x, y))

        return ordered_points
