import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
import time
from pyapriltags import Detector
from ipdb import set_trace
from scipy.spatial.transform import Rotation as R

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
    if len(points_drill) > 8192:
        pcd_object = o3d.geometry.PointCloud()
        pcd_object.points = o3d.utility.Vector3dVector(points_drill)
        pcd_object = pcd_object.farthest_point_down_sample(8192)
        points_drill = np.asarray(pcd_object.points)  # (8192, 3)

    # 用plotly可视化一下
    # plotly_vis_points(points_drill, title="Drill Points in Workspace")

    # open3d 可视化一下points
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_drill)
    # o3d.visualization.draw_geometries([pcd])
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_drill)
    # o3d.visualization.draw_geometries([pcd])

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
        # source_pcd.transform(driller_pose)
        # source_pcd.paint_uniform_color([1, 0, 0])
        # target_pcd.paint_uniform_color([0, 1, 0])
        # o3d.visualization.draw_geometries([source_pcd, target_pcd], window_name="Driller Pose Estimation")
            
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
    speed = 0.25 if args[0] else 0.5
    if distance > 0.01:
        velocity = direction / distance * speed  # 速度大小为 0.1 m/s
    else:
        velocity = np.array([0, 0, 0])
    action = np.array([velocity[0], velocity[1], 0]) if args[0] else np.array([-velocity[0], -velocity[1], 0])
    return action

def backward_quad_policy(current_quad_base_pose, initial_quad_base_pos, *args, **kwargs):
    """ 
    根据当前四足机器狗的基座位姿和目标初始位姿，计算倒退行动命令。
    此策略与 forward_quad_policy 逻辑类似，通过计算世界坐标系下的目标方向，
    并将其转换为机器狗局部坐标系下的速度指令。
    
    参数:
        current_quad_base_pose (np.ndarray): 机器狗当前在世界坐标系下的4x4齐次变换矩阵。
        initial_quad_base_pos (np.ndarray): 机器狗的初始重置位置（通常是 [x, y, z] 数组）。
    
    返回:
        np.ndarray: 四足机器狗的行动命令 [vx, vy, vz_rot]，其中 vz_rot 固定为 0。
    """
    # 提取机器狗当前在世界坐标系中的平移部分 (x, y, z)
    current_trans = current_quad_base_pose[:3, 3] 
    
    # 目标平移部分 (x, y, z)
    target_trans = initial_quad_base_pos 

    # 计算从当前位置到目标初始位置的方向向量
    direction = target_trans - current_trans
    
    # 计算2D平面上的距离（忽略Z轴，因为我们主要关心水平移动）
    distance_xy = np.linalg.norm(direction[:2]) # 只考虑XY平面距离

    speed = 0.25 if args[0] else 0.5  # m/s，机器狗的线速度，与前进策略保持一致，确保平稳。
    angular_speed = 0.0 # 暂不考虑旋转，保持机器狗方向不变，专注于倒退。

    # 如果机器狗已经非常接近目标位置，则停止运动，避免来回抖动。
    if distance_xy < 0.01: # 设置停止阈值
        velocity_world = np.array([0.0, 0.0, 0.0])
        cprint(f"已接近初始位置，距离: {distance_xy:.3f}m。停止运动！", 'green')
    else:
        # 计算在世界坐标系中的期望线速度向量
        # 这里的 direction 是从 current_trans 指向 target_trans
        velocity_world = direction / distance_xy * speed 
        
        # 将世界坐标系下的速度转换为机器狗的局部坐标系指令
        # 鉴于 `forward_quad_policy` 中 "机器狗头朝-x所以加个负号" 的说明，
        # 我们假设机器狗的 `quad_command[0]` 控制的是其局部X轴方向的运动，
        # 且其局部X轴与世界-X轴大致对齐。
        # 因此，若想在世界X轴正方向移动，`quad_command[0]` 需为负值。
        # 若想在世界X轴负方向移动，`quad_command[0]` 需为正值。
        # 这里 `velocity_world[0]` 是世界X方向的期望速度。
        # 对应的 `quad_command[0]` 应是其反向。
        action = np.array([-velocity_world[0], -velocity_world[1], angular_speed]) if args[0] else np.array([velocity_world[0], velocity_world[1], angular_speed])
        
        cprint(f"倒退中... 当前距离目标: {distance_xy:.3f}m, 世界速度: [{velocity_world[0]:.3f}, {velocity_world[1]:.3f}], 机器狗指令: [{action[0]:.3f}, {action[1]:.3f}]", 'blue')

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
    lift_height = 0.20 # 稍微增大避免磕桌子
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

def binary_search_xy_coordinate(env, init_qpos, current_start_trans_xy, target_coord_val, fixed_coord_val_other_xy, fixed_z_height, fixed_rot, coord_idx, max_iter=25, ik_retries=20):
    """
    在给定固定 Z 高度、固定另一轴 XY 坐标和固定旋转姿态的情况下，
    沿单一直线坐标（X 或 Y）执行二分查找，以找到最远的可达点。

    参数:
        env: 仿真环境实例。
        init_qpos: IK 求解器的起始关节配置。
        current_start_trans_xy: 当前 XY 位置 (例如，[x, y])，用于确定当前搜索坐标的起始值。
                                 例如，如果搜索 X，这将提供起始 X。
        target_coord_val: 目标坐标值（例如，end_trans[0] 用于 X）。
        fixed_coord_val_other_xy: 另一个 XY 坐标的固定值（例如，如果搜索 X，则为固定的 Y 值）。
        fixed_z_height: IK 尝试时固定的 Z 高度。
        fixed_rot: IK 尝试时固定的旋转矩阵。
        coord_idx: 0 表示 X 坐标搜索，1 表示 Y 坐标搜索。
        max_iter: 二分查找的最大迭代次数。
        ik_retries: 每次 IK 调用尝试的重试次数。

    返回:
        tuple: (最终可达的坐标值, 最佳关节位置) 如果成功，否则 (None, None)。
    """
    start_val = current_start_trans_xy[coord_idx] # 获取当前搜索坐标的起始值
    
    low_alpha = 0.0
    high_alpha = 1.0
    best_alpha = 0.0
    best_qpos = None
    
    # cprint(f"    [二分查找 {['X','Y'][coord_idx]}轴] 从 {start_val:.3f} 开始，目标 {target_coord_val:.3f} "
    #        f"(固定 {'Y' if coord_idx==0 else 'X'} 在 {fixed_coord_val_other_xy:.3f}, Z 在 {fixed_z_height:.3f})", 'yellow')

    # 构建 IK 尝试时的目标平移向量模板
    target_trans_template = np.array([0.0, 0.0, fixed_z_height])
    if coord_idx == 0: # 搜索 X 轴，则 Y 轴固定
        target_trans_template[1] = fixed_coord_val_other_xy
    else: # 搜索 Y 轴，则 X 轴固定
        target_trans_template[0] = fixed_coord_val_other_xy

    for i in range(max_iter):
        mid_alpha = (low_alpha + high_alpha) / 2.0
        
        # 计算当前测试的坐标值
        test_coord_val = start_val + mid_alpha * (target_coord_val - start_val)
        
        # 构造完整的 3D 目标平移向量
        test_trans = target_trans_template.copy()
        test_trans[coord_idx] = test_coord_val # 更新当前搜索的坐标值

        try:
            success, qpos_test = env.humanoid_robot_model.ik(
                trans=test_trans,
                rot=fixed_rot, # 使用固定的旋转姿态
                init_qpos=init_qpos,
                retry_times=ik_retries,
                rot_tol=1,  # 不限制旋转
            )
            if success:
                best_alpha = mid_alpha
                best_qpos = qpos_test.copy()
                low_alpha = mid_alpha
                # cprint(f"      迭代 {i+1}: 测试 {['X','Y'][coord_idx]}={test_coord_val:.3f} 成功。尝试更远。", 'green') # 调试输出
            else:
                high_alpha = mid_alpha
                # cprint(f"      迭代 {i+1}: 测试 {['X','Y'][coord_idx]}={test_coord_val:.3f} 失败。尝试更近。", 'red') # 调试输出
        except Exception as e:
            # cprint(f"      迭代 {i+1}: IK 过程发生异常，测试 {['X','Y'][coord_idx]}={test_coord_val:.3f} 失败: {e}。", 'red') # 调试输出
            high_alpha = mid_alpha
            
    if best_qpos is None:
        # cprint(f"    [二分查找 {['X','Y'][coord_idx]}轴] 未能找到任何超出起点的可达点 (alpha=0)。", 'red') # 调试输出
        return None, None
        
    final_reachable_coord_val = start_val + best_alpha * (target_coord_val - start_val)
    cprint(f"    [二分查找 {['X','Y'][coord_idx]}轴] 最佳可达 {['X','Y'][coord_idx]}: {final_reachable_coord_val:.3f} (alpha={best_alpha:.3f})", 'blue')
    return final_reachable_coord_val, best_qpos

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, hold_seconds=1, xy_refinement_iterations=5) -> Optional[np.ndarray]:
    """
    规划机械臂从当前位置到投放位置的轨迹。
    本版本调整了水平移动时 X、Y 坐标的二分查找方式，采用迭代式的同时二分查找，
    以增大时间开销并得到离理想目标更近的结果。
    在水平移动和 Z 轴下降时保持 begin_rot 姿态。
    
    参数:
        env: WrapperEnv 仿真环境实例。
        begin_qpos: 机械臂当前关节位置 (np.ndarray)。
        begin_trans: 机械臂末端执行器当前世界坐标系平移 (np.ndarray)。
        begin_rot: 机械臂末端执行器当前世界坐标系旋转矩阵 (np.ndarray)。
        end_trans: 目标投放点世界坐标系平移 (np.ndarray)。
        end_rot: 目标投放点世界坐标系旋转矩阵 (np.ndarray)。
        steps: 整个移动阶段的总步数 (int)。
        hold_seconds: 保持最终位置的秒数。
        xy_refinement_iterations: 对 XY 坐标进行迭代二分查找的次数，用于提高精度。
    
    返回:
        Optional[np.ndarray]: 轨迹，为一系列关节位置的 NumPy 数组，如果规划失败则为 None。
    """
    traj = []
    
    # 初始设置
    current_qpos_for_ik = begin_qpos.copy()
    current_eef_z_height = begin_trans[2] # 水平移动时维持此 Z 高度
    
    # 用于迭代二分查找的当前可达 XY 坐标
    current_reachable_x = begin_trans[0]
    current_reachable_y = begin_trans[1]
    
    cprint("\n--- Phase 1 & 2: 迭代规划沿 XY 轴的水平移动 (保持 begin_rot) ---", 'blue')
    
    # 存储每次迭代的最终关节位置，用于生成轨迹
    intermediate_qposes = []

    # 进行多轮迭代二分查找，以同时优化 X 和 Y
    for iteration in range(xy_refinement_iterations):
        cprint(f"  --- XY 迭代优化轮次 {iteration + 1}/{xy_refinement_iterations} ---", 'cyan')
        
        prev_reachable_x = current_reachable_x
        prev_reachable_y = current_reachable_y

        # 首先尝试优化 Y 轴
        cprint(f"    [XY 迭代] 优化 Y 轴. 当前可达 XY: ({current_reachable_x:.3f}, {current_reachable_y:.3f})", 'yellow')
        found_y_val, qpos_after_y_search = binary_search_xy_coordinate(
            env=env,
            init_qpos=current_qpos_for_ik,
            current_start_trans_xy=np.array([current_reachable_x, current_reachable_y]),
            target_coord_val=end_trans[1],
            fixed_coord_val_other_xy=current_reachable_x, # Y 轴搜索时固定当前可达的 X 值
            fixed_z_height=current_eef_z_height,
            fixed_rot=begin_rot,
            coord_idx=1 # 搜索 Y 轴
        )
        
        if qpos_after_y_search is not None:
            current_reachable_y = found_y_val
            current_qpos_for_ik = qpos_after_y_search.copy()
            cprint(f"    [XY 迭代] Y 轴优化结果: Y={current_reachable_y:.3f}", 'green')
        else:
            cprint(f"    [XY 迭代] Y 轴优化失败，保持 Y={current_reachable_y:.3f}。可能是初始点就不可达或目标方向不可达。", 'red')

        # 接着尝试优化 X 轴
        cprint(f"    [XY 迭代] 优化 X 轴. 当前可达 XY: ({current_reachable_x:.3f}, {current_reachable_y:.3f})", 'yellow')
        found_x_val, qpos_after_x_search = binary_search_xy_coordinate(
            env=env,
            init_qpos=current_qpos_for_ik, # 使用 Y 轴搜索后的关节位置作为起始
            current_start_trans_xy=np.array([current_reachable_x, current_reachable_y]),
            target_coord_val=end_trans[0],
            fixed_coord_val_other_xy=current_reachable_y, # X 轴搜索时固定当前可达的 Y 值 (已更新)
            fixed_z_height=current_eef_z_height,
            fixed_rot=begin_rot,
            coord_idx=0 # 搜索 X 轴
        )
        
        if qpos_after_x_search is not None:
            current_reachable_x = found_x_val
            current_qpos_for_ik = qpos_after_x_search.copy()
            cprint(f"    [XY 迭代] X 轴优化结果: X={current_reachable_x:.3f}", 'green')
        else:
            cprint(f"    [XY 迭代] X 轴优化失败，保持 X={current_reachable_x:.3f}。可能是初始点就不可达或目标方向不可达。", 'red')

        # 记录本次迭代的最终可达关节位置
        if current_qpos_for_ik is not None:
            intermediate_qposes.append(current_qpos_for_ik.copy())
            
        # 检查收敛性：如果 X 和 Y 坐标的变化量都非常小，则认为收敛
        distance_change_x = abs(current_reachable_x - prev_reachable_x)
        distance_change_y = abs(current_reachable_y - prev_reachable_y)
        convergence_threshold = 0.001 # 1毫米的阈值
        
        if iteration > 0 and distance_change_x < convergence_threshold and distance_change_y < convergence_threshold:
            cprint(f"  [XY 迭代] X/Y 坐标已收敛 (X 变化: {distance_change_x:.4f}, Y 变化: {distance_change_y:.4f})。", 'green')
            break
    
    # 如果在任何迭代后，仍然没有找到一个有效的关节位置，则规划失败
    if not intermediate_qposes:
        cprint("[PLAN_MOVE] 经过 XY 迭代二分查找，未能找到任何可达的关节位置。规划停止。", 'red')
        return None

    # 从 begin_qpos 移动到最终迭代的可达关节位置
    final_reachable_qpos = intermediate_qposes[-1]
    
    # 计算实际移动到最终可达点的轨迹步数
    steps_xy_move = max(10, steps // 2) # 分配一半的步数给 XY 移动
    traj_xy_move = plan_move_qpos(begin_qpos, final_reachable_qpos, steps=steps_xy_move)
    traj.extend(traj_xy_move)
    current_qpos_for_next_phase = final_reachable_qpos.copy() # 更新当前关节位置

    # --- Phase 3: 保持稳定阶段 ---
    # 在完成所有移动后，保持最终姿态一段时间，让机械臂稳定。
    if hold_seconds > 0:
        sim_dt = 0.02 # 假设环境步长，如果 env.config.ctrl_dt 可用，请使用环境配置
        try:
            # 尝试从环境中获取实际的步长，如果环境对象有这个属性的话
            sim_dt = env.config.ctrl_dt
        except Exception:
            cprint("无法从 env 获取 ctrl_dt，将使用默认dt。", 'yellow')

        hold_steps = max(1, int(hold_seconds / sim_dt)) # 确保至少 1 步
        cprint(f"\n--- Phase 3: 增加 {hold_seconds:.2f} 秒 ({hold_steps} 步) 稳定等待时间 ---", 'blue')
        # 将最后的关节位置重复 hold_steps 次，实现保持效果
        for _ in range(hold_steps):
            traj.append(current_qpos_for_next_phase)

    # 最终检查生成的轨迹是否为空
    if not traj:
        cprint("[PLAN_MOVE] 生成的轨迹为空。规划存在问题。", 'red')
        return None
        
    cprint(f"[PLAN_MOVE] 成功规划总移动轨迹，共 {len(traj)} 步。", 'green')
    return np.array(traj)

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
        forward_steps = 5000
        steps_per_camera_shot = 5
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0], head_camera_matrix[1, 1], head_camera_matrix[0, 2], head_camera_matrix[1, 2])
        target_container_pose = np.array([0.65, 0.0, 0.0])
        
        # Proportional control gains
        Kp_linear = 0.5  # For head tracking (unchanged)
        Kp_angular = 50  # For turning control
        
        def is_close(current_pose, target_pose, threshold=0.05):
            distance = np.linalg.norm(current_pose[:2] - target_pose[:2])
            return distance < threshold
        
        initial_quad_reset_pose = None
        turning_steps_remaining = 0  # Track remaining turning steps
        turned = False

        for step in range(forward_steps):
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

                # Head tracking logic (unchanged)
                current_head_qpos = env.sim.humanoid_head_qpos
                if marker_center is not None:
                    delta_x = marker_center[0] - image_center[0]
                    delta_y = marker_center[1] - image_center[1]
                    fx = head_camera_matrix[0, 0]
                    fy = head_camera_matrix[1, 1]
                    angle_delta_x = np.arctan(delta_x / fx)
                    angle_delta_y = np.arctan(delta_y / fy)
                    head_qpos_adjustment = Kp_linear * np.array([-angle_delta_x, angle_delta_y])
                    new_head_qpos = current_head_qpos + head_qpos_adjustment
                else:
                    new_head_qpos = current_head_qpos

                if trans_marker_world is not None:
                    trans_container_world = rot_marker_world @ np.array([0, -0.31, 0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    if initial_quad_reset_pose is None:
                        initial_quad_reset_pose = pose_container_world
                        initial_quad_reset_pos = trans_container_world
                    env.sim.debug_vis_pose(pose_container_world, mocap_id='debug_axis_1')
                else:
                    pose_container_world = None

                if pose_container_world is not None:
                    if turning_steps_remaining > 0:
                        # In turning phase: optimize pose[0, 1] to 1
                        orientation_error = 1.0 - pose_container_world[0, 1]  # Target pose[0, 1] = 1
                        angular_vel = Kp_angular * orientation_error  # Proportional control
                        angular_vel = np.clip(angular_vel, -1.0, 1.0)  # Limit angular velocity
                        quad_command = np.array([0, 0, angular_vel])
                        turning_steps_remaining -= 1
                        cprint(f"[Step {step}] Turning: {turning_steps_remaining} steps left, pose[0, 1]={pose_container_world[0, 1]:.3f}, angular_vel={angular_vel:.3f}", 'green')
                        if turning_steps_remaining == 0:
                            turned = True
                            cprint("[INFO] Completed 200-step turning phase.", 'green')
                    else:
                        # Check if close to target to initiate turning
                        if not turned and abs(pose_container_world[0, 3] - target_container_pose[0]) < 0.4:
                            cprint("[INFO] Almost reached target container position, starting 200-step turn.", 'green')
                            turning_steps_remaining = 200  # Start fixed 200-step turning phase
                            quad_command = np.array([0, 0, 0])  # Initialize to zero, will be set in next iteration
                        elif is_close(pose_container_world[:3, 3], target_container_pose, threshold=0.05):
                            cprint("[INFO] Close enough, jump out of the loop.", 'green')
                            quad_command = np.array([0, 0, 0])
                            break
                        else:
                            quad_command = forward_quad_policy(pose_container_world, target_container_pose, turned)
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
        """ trick: 翻转用于抓取的物体坐标系，使得旋转的x轴指向x轴正方向 """
        if args.obj == 'power_drill' and sum(obj_pose[:3, 0]) < 0:
            cprint("[INFO] Detected power drill with flipped x-axis, flipping the pose.", 'yellow')
            # 绕z轴旋转180度
            obj_pose[:3, :3] = obj_pose[:3, :3] @ np.diag([-1, -1, 1])
            env.sim.debug_vis_pose(obj_pose, mocap_id='debug_axis_4')  # 可视化翻转后的物体姿态
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
        # set_trace()   
    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        cprint("\n--- Starting Step 4: Plan to move and drop ---", 'yellow')

        # 获取机械臂当前末端执行器姿态（抓取并抬起后的姿态）
        # 'lift_plan' 是在 step 3 中生成的，其最后一个点就是机械臂抓取并抬起后的位置
        if grasp_plan is None or not lift_plan.shape[0] > 0:
            cprint("[ERROR] Grasp or lift plan is not valid. Cannot proceed to drop.", 'red')
            env.close()
            return
            
        current_arm_qpos = lift_plan[-1]
        current_eef_trans, current_eef_rot = env.humanoid_robot_model.fk_eef(current_arm_qpos)
        cprint(f"Current EEF (End Effector) pose after lift: Trans={current_eef_trans}, Rot={current_eef_rot}", 'cyan')

        # 确定放置目标姿态
        # trans_container_world 和 rot_container_world 是在 step 1 中获得的容器姿态
        if 'trans_container_world' not in locals() or trans_container_world is None:
            cprint("[ERROR] Container pose (trans_container_world) not detected in Step 1. Cannot plan drop.", 'red')
            env.close()
            return
            
        drop_height_above_container = 0  # 理想情况是容器上方0cm
        drop_target_trans = trans_container_world.copy()
        drop_target_trans[2] += drop_height_above_container # 增加Z轴高度

        # 放置时的旋转：可以保持当前抓取姿态的旋转，也可以设定一个固定的垂直姿态
        # 例如，如果希望物体垂直掉落，可以设置为单位矩阵（如果钻头主轴与Z轴对齐）
        # 这里先保持抓取时的旋转
        drop_target_rot = current_eef_rot.copy()
        # 如果需要特定放置方向，可以这样修改：
        # drop_target_rot = R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix() # 使其正面朝前
        
        cprint(f"Drop target pose: Trans={drop_target_trans}, Rot={drop_target_rot}", 'cyan')
        # 可视化放置目标点
        env.sim.debug_vis_pose(to_pose(drop_target_trans, drop_target_rot), mocap_id='debug_axis_3')

        # 规划移动轨迹
        move_plan_total_steps = 100  # 根据需求调整总步数
        move_plan = plan_move(
            env=env,
            begin_qpos=current_arm_qpos,
            begin_trans=current_eef_trans,
            begin_rot=current_eef_rot,
            end_trans=drop_target_trans,
            end_rot=drop_target_rot,
            steps=move_plan_total_steps,
            hold_seconds=1
        ) 
        
        if move_plan is None:
            cprint("[ERROR] Failed to plan move to drop position. Simulation halted.", 'red')
            env.close()
            return

        cprint("Executing move to drop position...", 'green')
        execute_plan(env, move_plan)
        
        cprint("Opening gripper to drop object...", 'green')
        open_gripper(env, steps=50) # 增加步数，确保夹爪完全打开
        cprint("Drop completed.", 'green')


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        cprint("\n--- 开始执行 Step 5: 机器狗倒退回初始位置 ---", 'yellow')
        cprint(f"目标初始位置: ({initial_quad_reset_pos[0]:.3f}, {initial_quad_reset_pos[1]:.3f}, {initial_quad_reset_pos[2]:.3f})", 'cyan')

        def is_quad_close_to_initial(current_quad_pose_matrix, target_initial_pos_vector, threshold=0.01):
            current_pos_xy = current_quad_pose_matrix[:2, 3]
            target_pos_xy = target_initial_pos_vector[:2]
            distance = np.linalg.norm(current_pos_xy - target_pos_xy)
            return distance < threshold

        backward_steps = 1000
        turning_steps_remaining = 0  # Track remaining turning steps
        turned = False

        for step in range(backward_steps):
            obs_head = env.get_obs(camera_id=0)
            # env.debug_save_obs(obs_head, runtime_name=runtime_name, step=step)
            trans_marker_world, rot_marker_world, marker_center = detect_marker_pose(
                detector,
                obs_head.rgb,
                head_camera_params,
                obs_head.camera_pose,
                tag_size=0.12
            )
            if trans_marker_world is not None:
                trans_container_world = rot_marker_world @ np.array([0, -0.31, 0.02]) + trans_marker_world
                rot_container_world = rot_marker_world
                pose_container_world = to_pose(trans_container_world, rot_container_world)
                current_quad_base_pose = pose_container_world
                env.sim.debug_vis_pose(pose_container_world, mocap_id='debug_axis_1')
            else:
                current_quad_base_pose = None

            if current_quad_base_pose is not None:
                if is_quad_close_to_initial(current_quad_base_pose, initial_quad_reset_pos):
                    cprint(f"机器狗已成功返回初始位置附近，总共用时 {step} 步。", 'green')
                    env.step_env(quad_command=np.array([0.0, 0.0, 0.0]))
                    break
                if turning_steps_remaining > 0:
                    # In turning phase: optimize pose[0, 1] to -1
                    orientation_error = -1.0 - current_quad_base_pose[0, 1]  # Target pose[0, 1] = -1
                    angular_vel = Kp_angular * orientation_error  # Proportional control
                    angular_vel = np.clip(angular_vel, -1.0, 1.0)  # Limit angular velocity
                    quad_command = np.array([0, 0, angular_vel])
                    turning_steps_remaining -= 1
                    cprint(f"[Step {step}] Turning: {turning_steps_remaining} steps left, pose[0, 1]={current_quad_base_pose[0, 1]:.3f}, angular_vel={angular_vel:.3f}", 'green')
                    if turning_steps_remaining == 0:
                        turned = True
                        cprint("[INFO] Completed 200-step turning phase.", 'green')
                else:
                    if not turned and abs(current_quad_base_pose[0, 3] - target_container_pose[0]) > 0.55:
                        cprint("[INFO] Almost left target position, starting 200-step turn.", 'green')
                        turning_steps_remaining = 200  # Start fixed 200-step turning phase
                        quad_command = np.array([0, 0, 0])  # Initialize to zero, will be set in next iteration
                    else:
                        quad_command = backward_quad_policy(current_quad_base_pose, initial_quad_reset_pos, turned)
            else:
                quad_command = np.array([0, 0, 0])

            env.step_env(quad_command=quad_command)

            if step % 100 == 0:
                if current_quad_base_pose is not None:
                    current_quad_pos_xy = current_quad_base_pose[:2, 3]
                    distance_to_target = np.linalg.norm(current_quad_pos_xy - initial_quad_reset_pos[:2])
                    cprint(f"Step {step}/{backward_steps}: 机器狗当前XY: ({current_quad_pos_xy[0]:.3f}, {current_quad_pos_xy[1]:.3f}), 距离初始点: {distance_to_target:.3f}m。", 'blue')
                else:
                    cprint(f"Step {step}/{backward_steps}: 未检测到当前位姿。", 'yellow')
        else:
            cprint(f"警告: 机器狗在 {backward_steps} 步内未能完全返回初始位置。", 'yellow')

        Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()
        cprint(f"更新: 四足机器狗返回状态: {Metric['quad_return']}", 'green')

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()
