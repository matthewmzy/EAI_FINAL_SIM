import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
import time
from pyapriltags import Detector
from ipdb import set_trace

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
from termcolor import cprint
from PIL import Image

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

import open3d as o3d
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go

def plotly_vis_points(points: np.ndarray, title: str = "Point Cloud"):
    """Visualize a point cloud using Plotly."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue')
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    fig.show()

def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    # [0.5,0.3,0.75]
    # pc_mask = (
    #     (pc[:, 0] > 0.3)
    #     & (pc[:, 0] < 0.65)
    #     & (pc[:, 1] > 0.15)
    #     & (pc[:, 1] < 0.5)
    #     & (pc[:, 2] > 0.7)
    #     & (pc[:, 2] < 0.9)
    # )
    pc_mask = (
        (pc[:, 0] > 0.2)
        & (pc[:, 1] > 0.2)
        & (pc[:, 2] > 0.5)
        & (pc[:, 2] < 0.9)
    )
    return pc_mask

def detect_driller_pose(img, depth, camera_matrix, camera_pose, pose_est_method, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    H, W = 720, 1280
    
    # # 保存图像
    # pil_img = Image.fromarray(img)
    # pil_img.show()

    # # 加载YOLO模型
    # set_trace()
    # model = YOLO("my_own_scripts/yolo11l-seg.pt")
    # results = model(pil_img)

    # # 加载sam模型
    # checkpoint = "my_own_scripts/sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "my_own_scripts/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     predictor.set_image(img)
    #     masks, _, _ = predictor.predict()

    # 调整输入到目标分辨率
    if (img.shape[0], img.shape[1]) != (H, W):
        img = cv2.resize(img, (W, H))
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]

    # 生成网格坐标
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    z = depth[v, u]  # (H, W)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 过滤无效点（深度<=0）
    valid_mask = z > 0
    points_camera = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)  # (N,3)
    # shuffle points
    np.random.shuffle(points_camera)
    # plotly_vis_points(points_camera[:10000], title="Camera Points")  # 可视化前10000个点
    points_world = np.einsum("ab,nb->na", camera_pose[:3, :3], points_camera) + camera_pose[:3, 3] # (N,3)
    # plotly_vis_points(points_world[:10000], title="World Points")  # 可视化前10000个点

    points_drill = points_world[get_workspace_mask(points_world)]  # 过滤到工作空间内的点

    # 使用RANSAC进行平面分割以移除桌面点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_drill)
    
    # 使用RANSAC检测桌面平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,  # 平面最大距离阈值（根据噪声调整）
                                            ransac_n=3,             # 拟合平面所需的最少点数
                                            num_iterations=1000)    # RANSAC迭代次数
    
    # 提取非平面点（即物体点）
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    points_drill = np.asarray(outlier_cloud.points)  # (M, 3)，钻头的点云
    
    # 可选：如果需要，进一步降采样物体点
    if len(points_drill) > 2048:
        pcd_object = o3d.geometry.PointCloud()
        pcd_object.points = o3d.utility.Vector3dVector(points_drill)
        pcd_object = pcd_object.farthest_point_down_sample(2048)
        points_drill = np.asarray(pcd_object.points)  # (2048, 3)

    # 用plotly可视化一下
    # plotly_vis_points(points_drill, title="Drill Points in Workspace")

    # open3d 可视化一下points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_drill)
    o3d.visualization.draw_geometries([pcd])

    if pose_est_method == "registration":
        """ 点云配准 """
        def preprocess_point_cloud(pcd, voxel_size):
            """降采样点云并计算法向量和 FPFH 特征"""
            pcd_down = pcd.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=50))
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=150))
            return pcd_down, pcd_fpfh
        def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
            """执行基于特征的 RANSAC 全局配准"""
            distance_threshold = voxel_size * 1.5
            print(f":: RANSAC registration with distance threshold {distance_threshold:.3f}")
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
            )
            return result
        full_pcd = np.load("asset/obj/power_drill/single_pcd.npy")
        obs_pcd = points_drill
        voxel_size = 0.003
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(full_pcd)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(obs_pcd)
        source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

        # 执行全局配准
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

        # 提取变换矩阵
        driller_pose = result_ransac.transformation

        # 可视化obs_pcd和配准后的full_pcd
        source_pcd.transform(driller_pose)
        source_pcd.paint_uniform_color([1, 0, 0])
        target_pcd.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Driller Pose Estimation")
            
    else:
        """ 用作业二估pose """
        from assignment2_zzh.src.model.est_coord import EstCoordNet
        from assignment2_zzh.src.path import get_exp_config_from_checkpoint
        from assignment2_zzh.src.config import Config

        CANONICAL_TRANS = np.array([-0.5, -0.25, -0.13])

        config = Config.from_yaml(get_exp_config_from_checkpoint(args[0]))
        model = EstCoordNet(config)
        model.load_state_dict(torch.load(args[0], map_location='cpu')['model'])
        model.eval().to(args[1])
        
        canonical_points = points_drill + CANONICAL_TRANS

        with torch.no_grad():
            # pred_trans, pred_rot = model.est(points[np.newaxis, ...].astype(np.float32).to(args[1]))
            pred_trans, pred_rot = model.est(torch.from_numpy(canonical_points[np.newaxis, ...]).to(args[1]).float())
            pred_trans = pred_trans[0]
            pred_rot = pred_rot[0]

        driller_pose = np.eye(4)
        driller_pose[:3, :3] = pred_rot.cpu().numpy()  # (3,3)
        driller_pose[:3, 3] = pred_trans.cpu().numpy() - CANONICAL_TRANS  # (3,)
 
    return driller_pose

def detect_marker_pose(
        detector: Detector,
        img: np.ndarray,
        camera_params: tuple,
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float]]]:
    """Detect AprilTag's world pose and image center position"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
    if len(tags) > 0:
        tag = tags[0]
        pose_marker_camera = np.eye(4)
        pose_marker_camera[:3, :3] = tag.pose_R
        pose_marker_camera[:3, 3] = tag.pose_t.squeeze()
        pose_marker_world = camera_pose @ pose_marker_camera
        trans_marker_world = pose_marker_world[:3, 3]
        rot_marker_world = pose_marker_world[:3, :3]
        center = tag.center  # Marker center in image (x, y) in pixels
        return trans_marker_world, rot_marker_world, center
    return None, None, None

def forward_quad_policy(current_pose, target_pose, *args, **kwargs):
    """根据当前盒子位姿和目标位姿计算四足机器人命令"""
    current_trans = current_pose[:3, 3]
    target_trans = target_pose
    direction = target_trans - current_trans
    distance = np.linalg.norm(direction)
    if distance > 0.01:
        velocity = direction / distance * 0.5  # 速度大小为 0.1 m/s
    else:
        velocity = np.array([0, 0, 0])
    action = np.array([-velocity[0], -velocity[1], 0])  # xy 平面速度，角速度为 0, 机器狗头朝-x所以加个负号
    return action

def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    action = np.array([0,0,0])
    return action

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""

    reach_steps = grasp_config.get('reach_steps', 35)
    lift_steps = grasp_config.get('lift_steps', 20)
    squeeze_steps = grasp_config.get('squeeze_steps', 10)
    approach_steps = grasp_config.get('approach_steps', 20)
    descent_steps = grasp_config.get('descent_steps', 15)
    approach_height = grasp_config.get('approach_height', 0.10)
    max_joint_change = grasp_config.get('max_joint_change', 0.3)
    
    # 获取当前机器人状态
    current_qpos = env.get_state()[:7]
    
    # 考虑夹爪深度调整抓取位置
    gripper_depth = 0.012
    adjusted_grasp_trans = grasp.trans - gripper_depth * grasp.rot[:, 0]
    target_rot = grasp.rot
    
    # 计算两个接近点
    # 第一个目标点：物体上方指定高度处
    approach_trans = adjusted_grasp_trans.copy()
    approach_trans[2] += approach_height
    
    # 验证两个目标点的IK可行性
    try:
        # 检查上方接近点的IK
        success_approach, approach_qpos = env.humanoid_robot_model.ik(
            trans=approach_trans,
            rot=target_rot,
            init_qpos=current_qpos,
            retry_times=5
        )
        if not success_approach:
            print("Approach point IK failed")
            return None
            
        # 检查最终抓取点的IK
        success_grasp, grasp_arm_qpos = env.humanoid_robot_model.ik(
            trans=adjusted_grasp_trans,
            rot=target_rot,
            init_qpos=approach_qpos,
            retry_times=5
        )
        if not success_grasp:
            print("Grasp point IK failed")
            return None
    except Exception:
        return None
    
    # 阶段1: 接近物体上方approach_height处
    traj_reach_approach = []
    
    # 获取当前末端执行器位姿
    current_trans, current_rot = env.humanoid_robot_model.fk_eef(current_qpos)
    joint_distance_approach = np.linalg.norm(approach_qpos - current_qpos)
    cartesian_distance_approach = np.linalg.norm(approach_trans - current_trans)
    
    print(f"Approach phase - Joint distance: {joint_distance_approach:.3f}, Cartesian distance: {cartesian_distance_approach:.3f}")
    print(f"Using {approach_steps} steps for approach phase")
    
    if joint_distance_approach > 3.0:  # 关节距离过大的阈值
        print("Joint distance too large, using multi-stage progressive strategy")
        
        # 采用更细粒度的分阶段策略
        # 将大的关节运动分解为多个小的安全步骤
        current_stage_qpos = current_qpos.copy()
        total_stages = max(int(joint_distance_approach / 1.0), 4)  # 至少分4个阶段
        
        print(f"Dividing motion into {total_stages} progressive stages")
        
        for stage in range(total_stages):
            # 计算这个阶段的目标关节位置
            stage_alpha = (stage + 1) / total_stages
            stage_target_qpos = current_qpos + stage_alpha * (approach_qpos - current_qpos)
            
            # 计算这个阶段需要的步数
            stage_joint_distance = np.linalg.norm(stage_target_qpos - current_stage_qpos)
            stage_steps = max(int(stage_joint_distance / 0.2) + 5, 8)  # 根据距离计算步数，最少8步
            
            print(f"Stage {stage+1}/{total_stages}: Joint distance {stage_joint_distance:.3f}, using {stage_steps} steps")
            
            # 生成这个阶段的轨迹
            for step in range(stage_steps):
                alpha = (step + 1) / stage_steps
                # 使用极其平滑的插值
                smooth_alpha = 10 * alpha**7 - 35 * alpha**6 + 84 * alpha**5 - 70 * alpha**4 + 20 * alpha**3  # 七次平滑插值
                interpolated_qpos = current_stage_qpos + smooth_alpha * (stage_target_qpos - current_stage_qpos)
                
                # 极其严格的关节变化控制
                if len(traj_reach_approach) > 0:
                    joint_change = np.linalg.norm(interpolated_qpos - traj_reach_approach[-1])
                    max_safe_change = max_joint_change * 0.5  # 更严格的限制
                    
                    if joint_change > max_safe_change:
                        # 进一步限制步长
                        safe_ratio = max_safe_change / joint_change
                        interpolated_qpos = traj_reach_approach[-1] + safe_ratio * (interpolated_qpos - traj_reach_approach[-1])
                        print(f"  Step limited: original change {joint_change:.3f} -> safe change {max_safe_change:.3f}")
                
                traj_reach_approach.append(interpolated_qpos.copy())
            
            # 更新当前阶段位置
            current_stage_qpos = stage_target_qpos.copy()
            
            # 阶段间短暂停顿，增加几个重复点确保稳定
            for _ in range(3):
                traj_reach_approach.append(current_stage_qpos.copy())
        
        print(f"Multi-stage approach completed with {len(traj_reach_approach)} total steps")
        
    elif joint_distance_approach > 1.5 or cartesian_distance_approach > 0.5:
        # 原有的大运动处理逻辑保持不变
        print("Large movement to approach point, using conservative interpolation")
        conservative_steps = max(approach_steps * 2, 40)  # 大运动时增加步数
        for i in range(conservative_steps):
            alpha = (i + 1) / conservative_steps
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # 三次平滑插值
            interpolated_qpos = current_qpos + smooth_alpha * (approach_qpos - current_qpos)
            traj_reach_approach.append(interpolated_qpos.copy())
    else:
        # 原有的笛卡尔空间规划逻辑保持不变
        print("Using Cartesian space planning for approach")
        for step in range(approach_steps):
            alpha = (step + 1) / approach_steps
            target_trans = current_trans + alpha * (approach_trans - current_trans)
            
            # 旋转插值
            from scipy.spatial.transform import Rotation as R, Slerp
            key_times = [0, 1]
            key_rotations = R.from_matrix(np.stack([current_rot, target_rot], axis=0))
            slerp = Slerp(key_times, key_rotations)
            r_interp = slerp(alpha)
            interp_rot = r_interp.as_matrix()
            
            try:
                success, new_qpos = env.humanoid_robot_model.ik(
                    trans=target_trans,
                    rot=interp_rot,
                    init_qpos=current_qpos if step == 0 else traj_reach_approach[-1],
                    retry_times=3
                )
                if success:
                    # 检查关节变化是否超过限制
                    prev_qpos = current_qpos if step == 0 else traj_reach_approach[-1]
                    joint_change = np.linalg.norm(new_qpos - prev_qpos)
                    if joint_change <= max_joint_change:
                        traj_reach_approach.append(new_qpos.copy())
                    else:
                        print(f"Joint change {joint_change:.3f} exceeds limit {max_joint_change:.3f}, using fallback")
                        # 使用关节空间插值作为备用
                        alpha_joint = (step + 1) / approach_steps
                        interpolated_qpos = current_qpos + alpha_joint * (approach_qpos - current_qpos)
                        traj_reach_approach.append(interpolated_qpos.copy())
                else:
                    # IK失败时使用关节空间插值
                    alpha_joint = (step + 1) / approach_steps
                    interpolated_qpos = current_qpos + alpha_joint * (approach_qpos - current_qpos)
                    traj_reach_approach.append(interpolated_qpos.copy())
            except Exception:
                # 异常时使用关节空间插值
                alpha_joint = (step + 1) / approach_steps
                interpolated_qpos = current_qpos + alpha_joint * (approach_qpos - current_qpos)
                traj_reach_approach.append(interpolated_qpos.copy())
    
    # 阶段2: 从上方竖直向下接近目标 (保持原有逻辑)
    traj_reach_descent = []
    
    print(f"Descent phase - planning {descent_steps} steps from approach to grasp")
    
    # 竖直下降轨迹规划
    for step in range(descent_steps):
        alpha = (step + 1) / descent_steps
        # 只在Z方向插值，保持X,Y位置不变
        target_trans = approach_trans.copy()
        target_trans[2] = approach_trans[2] + alpha * (adjusted_grasp_trans[2] - approach_trans[2])
        
        try:
            success, new_qpos = env.humanoid_robot_model.ik(
                trans=target_trans,
                rot=target_rot,
                init_qpos=approach_qpos if step == 0 else traj_reach_descent[-1],
                retry_times=3
            )
            if success:
                # 检查关节变化限制
                prev_qpos = approach_qpos if step == 0 else traj_reach_descent[-1]
                joint_change = np.linalg.norm(new_qpos - prev_qpos)
                if joint_change <= max_joint_change:
                    traj_reach_descent.append(new_qpos.copy())
                else:
                    # 使用更小的步长
                    alpha_joint = (step + 1) / descent_steps * 0.5  # 减半步长
                    interpolated_qpos = approach_qpos + alpha_joint * (grasp_arm_qpos - approach_qpos)
                    traj_reach_descent.append(interpolated_qpos.copy())
            else:
                # IK失败时使用关节空间插值
                alpha_joint = (step + 1) / descent_steps
                interpolated_qpos = approach_qpos + alpha_joint * (grasp_arm_qpos - approach_qpos)
                traj_reach_descent.append(interpolated_qpos.copy())
        except Exception:
            # 异常时使用关节空间插值
            alpha_joint = (step + 1) / descent_steps
            interpolated_qpos = approach_qpos + alpha_joint * (grasp_arm_qpos - approach_qpos)
            traj_reach_descent.append(interpolated_qpos.copy())
    
    # 合并接近轨迹
    traj_reach = traj_reach_approach + traj_reach_descent
    
    # 阶段3: 保持位置，夹取 (保持原有逻辑)
    traj_squeeze = []
    for i in range(squeeze_steps):
        traj_squeeze.append(grasp_arm_qpos.copy())
    
    # 阶段4: 抬起轨迹 (保持原有逻辑)
    traj_lift = []
    lift_height = 0.20 
    lift_target_trans = adjusted_grasp_trans.copy()
    lift_target_trans[2] += lift_height
    
    try:
        success, lift_end_qpos = env.humanoid_robot_model.ik(
            trans=lift_target_trans,
            rot=target_rot,
            init_qpos=grasp_arm_qpos,
            retry_times=5
        )
        
        if success:
            print(f"Lift IK successful, planning {lift_steps} step trajectory")
            for i in range(lift_steps):
                alpha = (i + 1) / lift_steps
                interpolated_qpos = grasp_arm_qpos + alpha * (lift_end_qpos - grasp_arm_qpos)
                
                # 检查关节变化限制
                prev_qpos = grasp_arm_qpos if i == 0 else traj_lift[-1]
                joint_change = np.linalg.norm(interpolated_qpos - prev_qpos)
                if joint_change <= max_joint_change:
                    traj_lift.append(interpolated_qpos.copy())
                else:
                    # 使用更小的步长
                    safe_alpha = min(alpha, alpha * (max_joint_change / joint_change))
                    safe_qpos = grasp_arm_qpos + safe_alpha * (lift_end_qpos - grasp_arm_qpos)
                    traj_lift.append(safe_qpos.copy())
        else:
            print("Lift end IK failed, using incremental planning")
            cur_trans = adjusted_grasp_trans.copy()
            cur_qpos = grasp_arm_qpos.copy()
            step_height = lift_height / lift_steps
            
            for step in range(lift_steps):
                cur_trans[2] += step_height
                try:
                    success, new_qpos = env.humanoid_robot_model.ik(
                        trans=cur_trans,
                        rot=target_rot,
                        init_qpos=cur_qpos,
                        retry_times=3
                    )
                    if success:
                        # 检查关节变化限制
                        joint_change = np.linalg.norm(new_qpos - cur_qpos)
                        if joint_change <= max_joint_change:
                            cur_qpos = new_qpos.copy()
                            traj_lift.append(cur_qpos.copy())
                        else:
                            # 使用渐进式小步移动
                            direction = new_qpos - cur_qpos
                            safe_step = direction * (max_joint_change / joint_change)
                            cur_qpos = cur_qpos + safe_step
                            traj_lift.append(cur_qpos.copy())
                    else:
                        if len(traj_lift) > 0:
                            prev_qpos = traj_lift[-1]
                            delta_qpos = (prev_qpos - grasp_arm_qpos) * 0.1
                            next_qpos = prev_qpos + delta_qpos
                            traj_lift.append(next_qpos.copy())
                        else:
                            traj_lift.append(grasp_arm_qpos.copy())
                except Exception:
                    if len(traj_lift) > 0:
                        traj_lift.append(traj_lift[-1].copy())
                    else:
                        traj_lift.append(grasp_arm_qpos.copy())
                        
    except Exception as e:
        print(f"Lift planning exception: {e}")
        print("Using fallback lift strategy")
        
        cur_trans = adjusted_grasp_trans.copy()
        cur_qpos = grasp_arm_qpos.copy()
        fallback_lift_height = 0.10  
        step_height = fallback_lift_height / lift_steps
        
        for step in range(lift_steps):
            cur_trans[2] += step_height
            try:
                success, new_qpos = env.humanoid_robot_model.ik(
                    trans=cur_trans,
                    rot=target_rot,
                    init_qpos=cur_qpos,
                    retry_times=1
                )
                if success:
                    joint_change = np.linalg.norm(new_qpos - cur_qpos)
                    if joint_change <= max_joint_change:
                        cur_qpos = new_qpos.copy()
                        traj_lift.append(cur_qpos.copy())
                    else:
                        direction = new_qpos - cur_qpos
                        safe_step = direction * (max_joint_change / joint_change)
                        cur_qpos = cur_qpos + safe_step
                        traj_lift.append(cur_qpos.copy())
                else:
                    if len(traj_lift) > 0:
                        prev_qpos = traj_lift[-1]
                        delta_qpos = (prev_qpos - grasp_arm_qpos) * 0.1
                        next_qpos = prev_qpos + delta_qpos
                        traj_lift.append(next_qpos.copy())
                    else:
                        traj_lift.append(grasp_arm_qpos.copy())
            except Exception:
                if len(traj_lift) > 0:
                    traj_lift.append(traj_lift[-1].copy())
                else:
                    traj_lift.append(grasp_arm_qpos.copy())
    
    # 确保轨迹长度
    while len(traj_lift) < lift_steps:
        if len(traj_lift) > 0:
            traj_lift.append(traj_lift[-1].copy())
        else:
            traj_lift.append(grasp_arm_qpos.copy())
    
    print(f"Planned trajectories - Reach: {len(traj_reach)} (approach: {len(traj_reach_approach)}, descent: {len(traj_reach_descent)}), Squeeze: {len(traj_squeeze)}, Lift: {len(traj_lift)}")
    
    # 检查轨迹有效性
    if len(traj_reach) == 0:
        return None
    return [np.array(traj_reach), np.array(traj_squeeze), np.array(traj_lift)]

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, *args, **kwargs):
    """Plan a trajectory moving the driller from table to dropping position"""
    # implement
    traj = []

    succ = False
    if not succ: return None
    return traj

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)
def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)
def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pose_est_method", type=str, default="registration", choices=["registration", "pointnet", "gt"])
    parser.add_argument("--est_drill_ckpt", type=str, default="assignment2_zzh/exps/exp2/checkpoint/checkpoint_15000.pth")


    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    
    head_init_qpos = np.array([0.0,0.0]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    
    observing_qpos = humanoid_init_qpos + np.array([0.,-0.3,-0.1,0.5,0.2,0.1,-0.5]) # you can customize observing qpos to get wrist obs
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps = 20)
    execute_plan(env, init_plan)

    """ mzy - debug """
    # env.sim.debug_vis_pose(to_pose(np.array([0,0,0]), np.eye(3)))
    # env.sim.debug_vis_pose(to_pose(np.array([1,0,0]), np.eye(3)))
    # env.sim.debug_vis_pose(to_pose(np.array([0,1,0]), np.eye(3)))
    # env.sim.debug_vis_pose(to_pose(np.array([0,0,1]), np.eye(3)))
    # env.sim.debug_vis_pose(to_pose(np.array([0.5,0.3,0.75]),
    #                                 np.eye(3)), mocap_id='debug_axis_0')   

    runtime_name = time.strftime("%Y%m%d_%H%M%S")

    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1000
        steps_per_camera_shot = 5
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0], head_camera_matrix[1, 1], head_camera_matrix[0, 2], head_camera_matrix[1, 2])
        target_container_pose = np.array([0.7, 0.0, 0.0])
        
        # Proportional control gain
        Kp = 0.5

        def is_close(current_pose, target_pose, threshold=0.05):
            distance = np.linalg.norm(current_pose[:2] - target_pose[:2])
            return distance < threshold

        for step in range(forward_steps): # forward_steps
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0)
                # env.debug_save_obs(obs_head, runtime_name=runtime_name, step=step)  # Save head camera observation, DEBUG
                img_height, img_width = obs_head.rgb.shape[:2]
                image_center = (img_width / 2, img_height / 2)

                trans_marker_world, rot_marker_world, marker_center = detect_marker_pose(
                    detector,
                    obs_head.rgb,
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )

                # Head tracking logic
                current_head_qpos = env.sim.humanoid_head_qpos  # Last two joints are head joints
                if marker_center is not None:
                    # Calculate pixel deviation
                    delta_x = marker_center[0] - image_center[0]
                    delta_y = marker_center[1] - image_center[1]
                    # cprint(f"Marker center: {marker_center}, Image center: {image_center}", 'cyan')
                    # cprint(f"Delta x: {delta_x}, Delta y: {delta_y}", 'cyan')

                    # Convert pixel deviation to angular deviation using focal lengths
                    fx = head_camera_matrix[0, 0]
                    fy = head_camera_matrix[1, 1]
                    angle_delta_x = np.arctan(delta_x / fx)
                    angle_delta_y = np.arctan(delta_y / fy)

                    # Apply proportional control
                    head_qpos_adjustment = Kp * np.array([-angle_delta_x, angle_delta_y])  # Negative due to joint orientation
                    new_head_qpos = current_head_qpos + head_qpos_adjustment
                    # cprint(f"Current head qpos: {current_head_qpos}, New head qpos: {new_head_qpos}", 'cyan')
                else:
                    new_head_qpos = current_head_qpos  # No adjustment if marker not detected

                if trans_marker_world is not None:
                    trans_container_world = rot_marker_world @ np.array([0, -0.31, 0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    env.sim.debug_vis_pose(pose_container_world, mocap_id='debug_axis_1')  # Visualize the container pose
                else:
                    pose_container_world = None

                if pose_container_world is not None:
                    quad_command = forward_quad_policy(pose_container_world, target_container_pose)
                    if is_close(pose_container_world[:3, 3], target_container_pose, threshold=0.05):
                        cprint("[INFO] Close enough, jump out of the loop.", 'green')
                        break
                else:
                    quad_command = np.array([0, 0, 0])

                env.step_env(
                    humanoid_head_qpos=new_head_qpos,
                    quad_command=quad_command
                )


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        if args.pose_est_method == 'gt':
            driller_pose = env.get_driller_pose()
        else:
            driller_pose = detect_driller_pose(rgb, depth,
                                                wrist_camera_matrix,
                                                camera_pose,
                                                args.pose_est_method,
                                                args.est_drill_ckpt,
                                                args.device)
            cprint(driller_pose-env.get_driller_pose(), 'cyan')
        cprint(driller_pose, 'green')
        env.sim.debug_vis_pose(driller_pose, mocap_id='debug_axis_2') 

        # TODO: ma zhiyuan modified here, assume the driller_pose is detected correctly
        #driller_pose = np.array([[1,0,0,0.5],[0,1,0,0.3],[0,0,1,0.75],[0,0,0,1]])
        
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n]
        
        grasp_config = dict( 
            reach_steps=35,      # 接近步数 (增加5步提高稳定性)
            lift_steps=20,       # 抬起步数 (增加5步使抬起更平稳)
            squeeze_steps=10,     # 夹取步数 (保持不变)
            approach_steps=20,    # 到上方接近点的步数
            descent_steps=15,     # 竖直下降步数
            approach_height=0.10, # 上方接近高度 (10cm)
            max_joint_change=0.3, # 单步最大关节变化
        )

        successful_grasp = False
        grasp_plan = None
        
        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                successful_grasp = True
                print(f"Successfully planned grasp with {len(grasp_plan)} phases")
                break
                
        if not successful_grasp:
            print("No valid grasp plan found.")
            env.close()
            return
            
        # 执行三阶段抓取，根据plan_grasp的返回值调整
        if len(grasp_plan) == 3:
            reach_plan, squeeze_plan, lift_plan = grasp_plan
        else:
            reach_plan, lift_plan = grasp_plan
            squeeze_plan = np.array([reach_plan[-1]] * 8)  

        # 执行抓取序列
        print("Executing reach phase...")
        pregrasp_plan = plan_move_qpos(observing_qpos, reach_plan[0], steps=30)
        execute_plan(env, pregrasp_plan)
        # 确保夹爪打开
        open_gripper(env, steps=10)
        # 执行接近轨迹
        execute_plan(env, reach_plan)
        print("Executing squeeze phase...")
        # 夹取阶段：保持位置，关闭夹爪
        execute_plan(env, squeeze_plan)
        close_gripper(env, steps=15)
        print("Executing lift phase...")
        # 抬起阶段
        execute_plan(env, lift_plan)
        print("Grasp and lift completed")
        set_trace()   
    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        move_plan = plan_move(
            env=env,
        ) 
        execute_plan(env, move_plan)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 1000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            quad_command = backward_quad_policy()
            env.step_env(
                quad_command=quad_command
            )
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()