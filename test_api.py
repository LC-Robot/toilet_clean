from utils.integrated_toilet_analysis import IntegratedToiletAnalyzer
import cv2
from utils.realsense_d435 import RealsenseAPI
import numpy as np

def test_api1(analyzer, bgr_image, depth_raw):
    """测试API1: detect_toilet()"""
    print("\n" + "="*60)
    print("[测试 API1] detect_toilet()")
    print("="*60)
    
    result = analyzer.detect_toilet(bgr_image)
    print(f"检测结果: {result}")
    assert 'detected' in result
    assert 'confidence' in result
    assert 'toilet_area' in result
    
    if not result['detected']:
        print("未检测到马桶")
        return False
    
    print("✓ API1 测试通过")
    return True

def test_api2(analyzer, bgr_image, depth_raw):
    """测试API2: get_toilet_mask()"""
    print("\n" + "="*60)
    print("[测试 API2] get_toilet_mask()")
    print("="*60)
    
    toilet_mask = analyzer.get_toilet_mask(bgr_image, depth_raw)
    print(f"获取到马桶mask，形状: {toilet_mask.shape}")
    assert toilet_mask is not None
    assert len(toilet_mask.shape) == 2

    save_path = "api_toilet_mask.png"
    cv2.imwrite(save_path, toilet_mask)
    print(f"已保存到: {save_path}")
    print("✓ API2 测试通过")

def test_api3(analyzer, bgr_image, depth_raw):
    """测试API3: get_rim_mask()"""
    print("\n" + "="*60)
    print("[测试 API3] get_rim_mask()")
    print("="*60)
    
    rim_mask = analyzer.get_rim_mask(bgr_image, depth_raw)
    assert rim_mask is not None
    assert len(rim_mask.shape) == 2
    
    save_path = "api_rim_mask.png"
    cv2.imwrite(save_path, rim_mask)
    print(f"已保存到: {save_path}")
    print("✓ API3 测试通过")

def test_api4(analyzer, bgr_image, depth_raw):
    """测试API4: get_rim_bottom_left_point()"""
    print("\n" + "="*60)
    print("[测试 API4] get_rim_bottom_left_point()")
    print("="*60)
    
    bottom_left_result = analyzer.get_rim_bottom_left_point(bgr_image, depth_raw)
    
    if bottom_left_result['found']:
        print(f"X坐标: {bottom_left_result['x']}")
        print(f"Y坐标: {bottom_left_result['y']}")
        print(f"深度值: {bottom_left_result['depth']} mm")
        
        # 可视化最左下角点
        vis_image = bgr_image.copy()
        x, y = bottom_left_result['x'], bottom_left_result['y']
        
        rim_mask = analyzer.get_rim_mask(bgr_image, depth_raw)
        contours, _ = cv2.findContours(rim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)  
        cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)            

        save_path = 'api_bottom_left_point.jpg'
        cv2.imwrite(save_path, vis_image)
        print(f"已保存到: {save_path}")
        print("✓ API4 测试通过")
    else:
        print("未找到有效点")

def test_api5(analyzer, bgr_image, depth_raw):
    """测试API5: get_rim_skeleton()"""
    print("\n" + "="*60)
    print("[测试 API5] get_rim_skeleton()")
    print("="*60)
    
    skeleton_result = analyzer.get_rim_skeleton(bgr_image, depth_raw)

    if skeleton_result is not None:       
        # 保存mask
        skeleton_mask = skeleton_result['skeleton_mask']
        cv2.imwrite('api_skeleton_mask.png', skeleton_mask)
        
        # 创建彩色可视化
        vis_skeleton = bgr_image.copy()
        
        # 绘制rim_mask
        rim_mask = analyzer.get_rim_mask(bgr_image, depth_raw)
        rim_overlay = vis_skeleton.copy()
        rim_overlay[rim_mask > 0] = [255, 0, 0]
        vis_skeleton = cv2.addWeighted(vis_skeleton, 0.7, rim_overlay, 0.3, 0)
        
        # 绘制inner_mask
        inner_mask = analyzer._inner_mask
        inner_contours, _ = cv2.findContours(inner_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_skeleton, inner_contours, -1, (0, 255, 0), 1)
        
        # 绘制骨架
        skeleton_dilated = cv2.dilate(skeleton_mask, np.ones((3,3), np.uint8), iterations=1)
        vis_skeleton[skeleton_dilated > 0] = [0, 0, 255]  
        
        # 标记关键点
        if len(skeleton_result['points']) > 0:
            step = max(1, len(skeleton_result['points']) // 20)  
            for i in range(0, len(skeleton_result['points']), step):
                x, y = skeleton_result['points'][i]
                cv2.circle(vis_skeleton, (x, y), 3, (0, 255, 0), -1)  
        
        # 保存可视化结果
        cv2.imwrite('api_skeleton_visualization.jpg', vis_skeleton)
        print("已保存骨架mask和可视化结果")
        print("✓ API5 测试通过")
    else:
        print("骨架提取失败")

def test_api6(analyzer, bgr_image, depth_raw):
    """测试API6: get_stain_mask() - 不含深度信息"""
    print("\n" + "="*60)
    print("[测试 API6] get_stain_mask() - 不含深度信息")
    print("="*60)
    
    stain_mask = analyzer.get_stain_mask(
        bgr_image, depth_raw, 
        region_type='rim', 
        include_depth_info=False
    )
    assert stain_mask is not None
    assert len(stain_mask.shape) == 2

    save_path = "api_rim_stain_mask.png"
    cv2.imwrite(save_path, stain_mask)
    print(f"已保存到: {save_path}")
    print("✓ API6 测试通过")

def test_api7(analyzer, bgr_image, depth_raw):
    """测试API7: get_stain_mask() - 含深度信息"""
    print("\n" + "="*60)
    print("[测试 API7] get_stain_mask() - 含深度信息")
    print("="*60)
    
    stain_result = analyzer.get_stain_mask(
        bgr_image, depth_raw, 
        region_type='inner', 
        include_depth_info=True
    )
    assert stain_result is not None
    assert 'stain_mask' in stain_result
    assert 'depth_values' in stain_result
    assert 'depth_stats' in stain_result
    print(f"深度统计: {stain_result['depth_stats']}")

    save_path = "api_inner_stain_mask.png"
    cv2.imwrite(save_path, stain_result['stain_mask'])
    print(f"已保存到: {save_path}")
    print("✓ API7 测试通过")

def test_api8(analyzer, bgr_image, depth_raw):
    """测试API8: is_cleaned()"""
    print("\n" + "="*60)
    print("[测试 API8] is_cleaned()")
    print("="*60)
    
    clean_result = analyzer.is_cleaned(
        bgr_image, depth_raw, 
        region_type='inner', 
    )
    assert 'is_cleaned' in clean_result
    assert 'stain_count' in clean_result

    print(f"是否清洁: {clean_result['is_cleaned']}")
    print(f"污渍数量: {clean_result['stain_count']}")
    print("✓ API8 测试通过")

def test_api9(analyzer, bgr_image, depth_raw):
    """测试API9: get_stain_deepest_points()"""
    print("\n" + "="*60)
    print("[测试 API9] get_stain_deepest_points()")
    print("="*60)
    
    stains_info = analyzer.get_stain_deepest_points(bgr_image, depth_raw, region_type='inner')
    print(f"\n检测到 {len(stains_info)} 个污渍")
    
    for stain in stains_info:
        deepest = stain['deepest_point']
        center = stain['center_point']

        print(f"\n污渍 {stain['stain_id']}:")
        print(f"  最深点: 坐标({deepest['x']}, {deepest['y']}), 深度{deepest['depth']}mm")
        print(f"  中心点: 坐标({center['x']}, {center['y']}), 深度{center['depth']}mm")
        print(f"  污渍面积: {stain['stain_area']} 像素")

    # 可视化结果
    if len(stains_info) > 0:
        print("\n正在生成可视化结果...")
        vis_image = bgr_image.copy()
        
        # 获取inner mask并绘制轮廓
        inner_mask = analyzer.get_stain_mask(bgr_image, depth_raw, region_type='inner')
        if inner_mask is not None:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                inner_mask, connectivity=8
            )
            
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
            colors[0] = [0, 0, 0]  
            
            colored_mask = colors[labels]
            
            overlay = vis_image.copy()
            overlay[inner_mask > 0] = colored_mask[inner_mask > 0]
            vis_image = cv2.addWeighted(vis_image, 0.6, overlay, 0.4, 0)
        
        # 标记每个污渍的最深点和中心点
        for stain in stains_info:
            stain_id = stain['stain_id']
            deepest = stain['deepest_point']
            center = stain['center_point']
            
            # 绘制最深点（红色）
            cv2.circle(vis_image, (deepest['x'], deepest['y']), 1, (0, 0, 255), -1)
            
            # 绘制中心点（绿色）
            cv2.circle(vis_image, (center['x'], center['y']), 1, (0, 255, 0), -1)
            
            # 添加文字标注（在最深点旁边）
            text = f"#{stain_id}:D{deepest['depth']}mm"
            cv2.putText(vis_image, text, (deepest['x'] + 10, deepest['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (deepest['x'] + 10, deepest['y'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 添加中心点标注
            text_center = f"C{center['depth']}mm"
            cv2.putText(vis_image, text_center, (center['x'] + 10, center['y'] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis_image, text_center, (center['x'] + 10, center['y'] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 添加标题和图例
        title = f"Inner Stains Analysis: {len(stains_info)} stains detected"
        cv2.putText(vis_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 添加图例
        legend_y = 60
        cv2.circle(vis_image, (20, legend_y), 3, (0, 0, 255), -1)
        cv2.putText(vis_image, "Deepest Point", (30, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.circle(vis_image, (20, legend_y + 25), 3, (0, 255, 0), -1)
        cv2.putText(vis_image, "Center Point", (30, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 保存结果
        output_path = 'api9_inner_stains_deepest_points.jpg'
        cv2.imwrite(output_path, vis_image)
        print(f"已保存到: {output_path}")
    
    print("✓ API9 测试通过")


def test_api():
    """主测试函数"""
    print("="*60)
    print("初始化RealSense相机和图像采集")
    print("="*60)
    
    cam = RealsenseAPI()
    print(f"相机数量: {cam.get_num_cameras()}")
    
    print("正在获取图像数据...")
    color_img, depth_img = cam.read_frame()

    bgr_image = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    depth_raw = depth_img
    
    analyzer = IntegratedToiletAnalyzer(adaptive_ratio=0.6)
    
    try:
        # 测试API1：检测马桶
        # test_api1(analyzer, bgr_image, depth_raw)
        
        # 测试API2：获取马桶mask
        test_api2(analyzer, bgr_image, depth_raw)
        
        # 测试API3：获取rim mask
        test_api3(analyzer, bgr_image, depth_raw)
        
        # 测试API4：获取rim最左下角点
        test_api4(analyzer, bgr_image, depth_raw)
        
        # 测试API5：获取rim骨架
        test_api5(analyzer, bgr_image, depth_raw)
        
        # 测试API6：获取污渍mask（不含深度信息）
        # test_api6(analyzer, bgr_image, depth_raw)
        
        # 测试API7：获取污渍mask（含深度信息）
        test_api7(analyzer, bgr_image, depth_raw)
        
        # 测试API8：判断是否清洁
        # test_api8(analyzer, bgr_image, depth_raw)
        
        # 测试API9：获取污渍最深点和中心点
        test_api9(analyzer, bgr_image, depth_raw)
        
        print("\n" + "="*60)
        print("所有选定的测试已完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_api()
