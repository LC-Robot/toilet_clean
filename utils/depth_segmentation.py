#!/usr/bin/env python3

import cv2
import numpy as np
import os
from pathlib import Path


class ToiletDepthSegmentation:
    
    def __init__(self):
        self.rgb = None
        self.depth = None
        self.mask = None
        self.depth_stats = None
        self.rim_mask = None
        self.inner_mask = None
        self.drain_hole_mask = None  # 新增：排水口mask
        
    def load_data(self, rgb_path, depth_path, mask_path):
        self.rgb = cv2.imread(rgb_path)
        if self.rgb is None:
            return False
        
        self.depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if self.depth is None:
            return False
        
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            return False
        
        _, self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        
        return True
    
    def preprocess_depth(self, kernel_size=5):
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        temp_depth = np.zeros_like(self.depth, dtype=np.uint16)
        temp_depth[self.mask > 0] = self.depth[self.mask > 0]
        
        valid_depths = temp_depth[temp_depth > 0]
        if len(valid_depths) == 0:
            return
        
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        
        temp_depth_norm = ((temp_depth - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        
        smoothed_8bit = cv2.medianBlur(temp_depth_norm, kernel_size)
        
        smoothed_depth = ((smoothed_8bit / 255.0) * (max_depth - min_depth) + min_depth).astype(np.uint16)
        
        self.depth[self.mask > 0] = smoothed_depth[self.mask > 0]
    
    def analyze_depth_distribution(self):
        toilet_depth = self.depth[self.mask > 0]
        valid_depth = toilet_depth[toilet_depth > 0]
        
        if len(valid_depth) == 0:
            return False
        
        self.depth_stats = {
            'min': np.min(valid_depth),
            'max': np.max(valid_depth),
            'mean': np.mean(valid_depth),
            'median': np.median(valid_depth),
            'std': np.std(valid_depth),
            'p25': np.percentile(valid_depth, 25),
            'p75': np.percentile(valid_depth, 75),
            'p90': np.percentile(valid_depth, 90),
            'valid_depth': valid_depth
        }
        
        return True
    
    def segment_adaptive(self, adaptive_ratio=0.5):
        if self.depth_stats is None:
            return False
        
        threshold = self.depth_stats['p25'] + \
                   (self.depth_stats['p75'] - self.depth_stats['p25']) * adaptive_ratio
        
        self.rim_mask = np.zeros_like(self.mask)
        self.inner_mask = np.zeros_like(self.mask)
        
        toilet_region = self.mask > 0
        
        self.rim_mask[toilet_region & (self.depth < threshold) & (self.depth > 0)] = 255
        self.inner_mask[toilet_region & (self.depth >= threshold)] = 255
        
        total_pixels = np.count_nonzero(self.rim_mask) + np.count_nonzero(self.inner_mask)
        
        if total_pixels == 0:
            return False
        
        return True
    
    def segment_drain_hole(self, depth_percentile=95, circularity_threshold=0.6, 
                          min_area_ratio=0.01, max_area_ratio=0.15):
        """
        检测并分割马桶排水口区域
        """
        if self.inner_mask is None or self.depth is None:
            return False
        
        # 提取inner区域的深度值
        inner_depths = self.depth[self.inner_mask > 0]
        valid_inner_depths = inner_depths[inner_depths > 0]
        
        if len(valid_inner_depths) < 100:
            return False
        
        # 找到最深的区域
        depth_threshold = np.percentile(valid_inner_depths, depth_percentile)
        
        # 创建深坑候选mask
        drain_candidate = np.zeros_like(self.inner_mask)
        drain_candidate[(self.inner_mask > 0) & (self.depth >= depth_threshold)] = 255
        
        # 形态学处理：闭运算连接碎片
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        drain_candidate = cv2.morphologyEx(drain_candidate, cv2.MORPH_CLOSE, kernel)
        
        # 找到所有连通区域
        contours, _ = cv2.findContours(drain_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        # 计算inner区域总面积
        inner_area = np.count_nonzero(self.inner_mask)
        min_area = inner_area * min_area_ratio
        max_area = inner_area * max_area_ratio
        
        # 筛选符合条件的轮廓
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < min_area or area > max_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # 圆度过滤
            if circularity < circularity_threshold:
                continue
  
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # 计算inner区域的中心
            inner_coords = np.column_stack(np.where(self.inner_mask > 0))
            inner_center_y = np.mean(inner_coords[:, 0])
            inner_center_x = np.mean(inner_coords[:, 1])
            
            # 计算到中心的距离
            distance_to_center = np.sqrt((cx - inner_center_x)**2 + (cy - inner_center_y)**2)
            max_distance = np.sqrt(inner_area / np.pi)  # 假设inner区域为圆形的半径
            normalized_distance = distance_to_center / max_distance if max_distance > 0 else 1.0
            
            score = circularity * (1.0 - normalized_distance * 0.5)
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        # 如果找到了合适的排水口
        if best_contour is not None:
            self.drain_hole_mask = np.zeros_like(self.inner_mask)
            cv2.drawContours(self.drain_hole_mask, [best_contour], -1, 255, -1)
            
            EXPANSION_KERNEL_SIZE = 18  # 膨胀核大小（可调整：5, 10, 15, 20等）
            EXPANSION_ITERATIONS = 2    # 膨胀迭代次数（可调整：1, 2, 3等）
            
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                     (EXPANSION_KERNEL_SIZE, EXPANSION_KERNEL_SIZE))
            self.drain_hole_mask = cv2.dilate(self.drain_hole_mask, kernel_dilate, 
                                              iterations=EXPANSION_ITERATIONS)
            
            # 确保在inner区域内
            self.drain_hole_mask = cv2.bitwise_and(self.drain_hole_mask, self.inner_mask)
            
            return True
        
        return False
    
    def post_process(self):
        if self.rim_mask is None or self.inner_mask is None:
            return False
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        # 创建有效深度区域mask（排除深度为0的区域）
        valid_depth_mask = (self.depth > 0).astype(np.uint8) * 255
        # 限制在原始马桶mask内
        valid_region = cv2.bitwise_and(self.mask, valid_depth_mask)
        
        inner_cleaned = self._keep_largest_component(self.inner_mask)
        isolated = cv2.subtract(self.inner_mask, inner_cleaned)
        rim_corrected = cv2.bitwise_or(self.rim_mask, isolated)
        
        rim_processed = cv2.morphologyEx(rim_corrected, cv2.MORPH_CLOSE, kernel_medium)
        inner_processed = cv2.morphologyEx(inner_cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        rim_processed = cv2.morphologyEx(rim_processed, cv2.MORPH_OPEN, kernel_small)
        inner_processed = cv2.morphologyEx(inner_processed, cv2.MORPH_OPEN, kernel_small)
        
        rim_processed = self._fill_holes(rim_processed)
        inner_processed = self._fill_holes(inner_processed)
        
        inner_processed = self._keep_largest_component(inner_processed)
        
        overlap = cv2.bitwise_and(rim_processed, inner_processed)
        if np.count_nonzero(overlap) > 0:
            rim_processed = cv2.subtract(rim_processed, overlap)
        
        # 限制rim_mask和inner_mask只在有效深度区域内
        rim_processed = cv2.bitwise_and(rim_processed, valid_region)
        inner_processed = cv2.bitwise_and(inner_processed, valid_region)
        
        self.rim_mask = rim_processed
        self.inner_mask = inner_processed
        
        return True
    
    def _keep_largest_component(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return mask
        
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return mask
        
        largest_label = np.argmax(areas) + 1
        cleaned_mask = np.zeros_like(mask)
        cleaned_mask[labels == largest_label] = 255
        
        return cleaned_mask
    
    def _fill_holes(self, mask):
        mask_floodfill = mask.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_floodfill, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(mask_floodfill)
        filled = cv2.bitwise_or(mask, holes)
        return filled
    
    def visualize_and_save(self, output_dir='toilet_segmentation_output'):
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, 'mask_rim.png'), self.rim_mask)
        cv2.imwrite(os.path.join(output_dir, 'mask_inner.png'), self.inner_mask)
        
        # 如果有排水口mask，也保存
        if self.drain_hole_mask is not None:
            cv2.imwrite(os.path.join(output_dir, 'mask_drain_hole.png'), self.drain_hole_mask)
        
        overlay = self.rgb.copy()
        overlay[self.rim_mask > 0] = [255, 0, 0]  # Blue
        overlay[self.inner_mask > 0] = [0, 255, 0] # Green
        
        # 如果有排水口，用红色标记
        if self.drain_hole_mask is not None:
            overlay[self.drain_hole_mask > 0] = [0, 0, 255]  # Red
        
        result = cv2.addWeighted(self.rgb, 0.6, overlay, 0.4, 0)
        
        rim_contours, _ = cv2.findContours(self.rim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inner_contours, _ = cv2.findContours(self.inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(result, rim_contours, -1, (255, 0, 0), 3)
        cv2.drawContours(result, inner_contours, -1, (0, 255, 0), 3)
        
        # 如果有排水口，绘制轮廓
        if self.drain_hole_mask is not None:
            drain_contours, _ = cv2.findContours(self.drain_hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, drain_contours, -1, (0, 0, 255), 2)
        
        legend_y, legend_size = 30, 25
        cv2.rectangle(result, (10, legend_y-15), (10+legend_size, legend_y+10), (255, 0, 0), -1)
        cv2.putText(result, "Toilet Rim (坐垫)", (10+legend_size+10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        legend_y += 40
        cv2.rectangle(result, (10, legend_y-15), (10+legend_size, legend_y+10), (0, 255, 0), -1)
        cv2.putText(result, "Inner Bottom (内部)", (10+legend_size+10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.drain_hole_mask is not None:
            legend_y += 40
            cv2.rectangle(result, (10, legend_y-15), (10+legend_size, legend_y+10), (0, 0, 255), -1)
            cv2.putText(result, "Drain Hole (排水口)", (10+legend_size+10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(output_dir, 'segmentation_result.jpg'), result)
        
        depth_norm = cv2.normalize(self.depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_colored[self.mask == 0] = [0, 0, 0]
        cv2.imwrite(os.path.join(output_dir, 'depth_visualization.jpg'), depth_colored)
        
        fig_combined = np.zeros((self.rgb.shape[0], self.rgb.shape[1]*3, 3), dtype=np.uint8)
        fig_combined[:, :self.rgb.shape[1]] = self.rgb
        fig_combined[:, self.rgb.shape[1]:self.rgb.shape[1]*2] = result
        fig_combined[:, self.rgb.shape[1]*2:] = depth_colored
        cv2.imwrite(os.path.join(output_dir, 'comparison.jpg'), fig_combined)
        
        return True
    
    def get_statistics(self):
        if self.rim_mask is None or self.inner_mask is None:
            return None
        
        rim_pixels = np.count_nonzero(self.rim_mask)
        inner_pixels = np.count_nonzero(self.inner_mask)
        total_pixels = rim_pixels + inner_pixels
        
        rim_depths = self.depth[self.rim_mask > 0]
        rim_depths = rim_depths[rim_depths > 0]
        rim_avg_depth = np.mean(rim_depths) if len(rim_depths) > 0 else 0
        
        inner_depths = self.depth[self.inner_mask > 0]
        inner_depths = inner_depths[inner_depths > 0]
        inner_avg_depth = np.mean(inner_depths) if len(inner_depths) > 0 else 0
        
        return {
            'rim_pixels': rim_pixels,
            'inner_pixels': inner_pixels,
            'total_pixels': total_pixels,
            'rim_ratio': rim_pixels / total_pixels * 100 if total_pixels > 0 else 0,
            'inner_ratio': inner_pixels / total_pixels * 100 if total_pixels > 0 else 0,
            'rim_avg_depth': rim_avg_depth,
            'inner_avg_depth': inner_avg_depth,
            'depth_difference': inner_avg_depth - rim_avg_depth
        }
    
    def print_summary(self):
        pass


def segment_toilet_depth_complete(rgb, depth, mask, adaptive_ratio=0.5, 
                                   kernel_size=5, output_dir=None, verbose=True,
                                   detect_drain_hole=True, drain_depth_percentile=95):
    """
    完整的马桶深度分割流程
    
    Args:
        rgb: RGB图像 (np.ndarray) 或图像路径 (str)
        depth: 深度图像 (np.ndarray) 或图像路径 (str)
        mask: 马桶mask (np.ndarray) 或图像路径 (str)
        adaptive_ratio: 自适应阈值比例 (0.0-1.0)
        kernel_size: 预处理中值滤波核大小
        output_dir: 输出目录（可选，如果为None则不保存）
        verbose: 是否打印详细信息
        detect_drain_hole: 是否检测排水口
        drain_depth_percentile: 排水口深度百分位阈值（95表示最深的5%）
        
    Returns:
        dict or None
    """
    segmenter = ToiletDepthSegmentation()
    
    if isinstance(rgb, str):
        if not segmenter.load_data(rgb, depth, mask):
            if verbose: print("数据加载失败")
            return None
    else:
        segmenter.rgb = rgb.copy() if rgb is not None else None
        segmenter.depth = depth.copy() if depth is not None else None
        segmenter.mask = mask.copy() if mask is not None else None
        
        if segmenter.rgb is None or segmenter.depth is None or segmenter.mask is None:
            if verbose: print("数据无效")
            return None
            
        _, segmenter.mask = cv2.threshold(segmenter.mask, 127, 255, cv2.THRESH_BINARY)
    
    segmenter.preprocess_depth(kernel_size=kernel_size)
    
    if not segmenter.analyze_depth_distribution():
        if verbose: print("深度分析失败")
        return None
    
    if not segmenter.segment_adaptive(adaptive_ratio=adaptive_ratio):
        if verbose: print("分割失败")
        return None
    
    if not segmenter.post_process():
        if verbose: print("后处理失败")
        return None
    
    # 检测排水口
    drain_hole_detected = False
    if detect_drain_hole:
        drain_hole_detected = segmenter.segment_drain_hole(depth_percentile=drain_depth_percentile)
        if verbose and drain_hole_detected:
            drain_pixels = np.count_nonzero(segmenter.drain_hole_mask)
            print(f"  ✓ 检测到排水口: {drain_pixels} 像素")
    
    if output_dir is not None:
        if not segmenter.visualize_and_save(output_dir=output_dir):
            if verbose: print("保存失败")
            return None
    
    statistics = segmenter.get_statistics()
    
    if verbose:
        print("\n✓ 深度分割完成")
        print(f"  - 马桶坐垫: {statistics['rim_pixels']:,} 像素 ({statistics['rim_ratio']:.1f}%)")
        print(f"  - 马桶内部: {statistics['inner_pixels']:,} 像素 ({statistics['inner_ratio']:.1f}%)")
        print(f"  - 深度差: {statistics['depth_difference']:.1f} mm")
    
    return {
        'rim_mask': segmenter.rim_mask,
        'inner_mask': segmenter.inner_mask,
        'drain_hole_mask': segmenter.drain_hole_mask,
        'segmenter': segmenter,
        'statistics': statistics
    }


def main():
    rgb_path = '/home/le/clean_ws/realsense_output/capture_20251008_161010_0000.png'
    depth_path = '/home/le/clean_ws/realsense_output/capture_20251008_161010_0000_depth_raw.png'
    mask_path = '/home/le/clean_ws/mask1.png'
    output_dir = 'toilet_segmentation_output'
    
    segment_toilet_depth_complete(
        rgb=rgb_path,
        depth=depth_path,
        mask=mask_path,
        adaptive_ratio=0.5,
        kernel_size=5,
        output_dir=output_dir,
        verbose=False
    )
    

if __name__ == '__main__':
    main()