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
from assignment2_zzh.src.model.est_coord import EstCoordNet
from assignment2_zzh.src.path import get_exp_config_from_checkpoint
from assignment2_zzh.src.config import Config

import torch

import open3d as o3d
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objects as go

CANONICAL_TRANS = np.array([-0.5, -0.25, -0.13])

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
    pc_mask = (
        (pc[:, 0] > 0.3)
        & (pc[:, 0] < 0.65)
        & (pc[:, 1] > 0.15)
        & (pc[:, 1] < 0.5)
        & (pc[:, 2] > 0.74)
        & (pc[:, 2] < 0.9)
    )
    return pc_mask

def detect_driller_pose(img, depth, camera_matrix, camera_pose, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """

    # [-0.00126555 -0.04089614  0.62149937] is train data mean translation

    # implement the detection logic here
    # 
    H, W = 720, 1280
    
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
    plotly_vis_points(points_world[:10000], title="World Points")  # 可视化前10000个点

    points_drill = points_world[get_workspace_mask(points_world)]  # 过滤到工作空间内的点
    # 用open3d fps 降采样到1024个点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_drill)
    pcd = pcd.farthest_point_down_sample(1024)  # 降采样到1024个点
    points_drill = np.asarray(pcd.points)  # (1024, 3)

    # 用plotly可视化一下
    # plotly_vis_points(points_drill, title="Drill Points in Workspace")

    # open3d 可视化一下points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_drill)
    o3d.visualization.draw_geometries([pcd])

    # set_trace()
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

    reach_steps = grasp_config.get('reach_steps', 20)
    lift_steps = grasp_config.get('lift_steps', 15)
    squeeze_steps = grasp_config.get('squeeze_steps', 10)
    delta_dist = grasp_config.get('delta_dist', 0.03)
    # print(reach_steps)
    # 获取当前机器人状态
    current_qpos = env.get_state()[:7]
    
    # 考虑夹爪深度调整抓取位置
    gripper_depth = 0.02  
    adjusted_grasp_trans = grasp.trans - gripper_depth * grasp.rot[:, 0]
    target_rot = grasp.rot
    
    try:
        success, grasp_arm_qpos = env.humanoid_robot_model.ik(
            trans=adjusted_grasp_trans,
            rot=target_rot,
            init_qpos=current_qpos,
            retry_times=5
        )
        # print(grasp_arm_qpos)
        if not success:
            print("Not Success")
            return None
    except Exception:
        return None
    
    # 阶段1: 反向规划接近轨迹
    approach_trajectory = []
    cur_trans = adjusted_grasp_trans.copy()
    cur_rot = target_rot.copy()
    cur_qpos = grasp_arm_qpos.copy()
    
    # 从抓取位置开始，向后规划接近点
    valid_approach_points = [grasp_arm_qpos.copy()]  # 抓取位置作为最后一个点
    
    for step in range(reach_steps):
        # 沿着抓取方向反向移动
        cur_trans = cur_trans - delta_dist * cur_rot[:, 0]
        try:
            success, cur_qpos = env.sim.humanoid_robot_model.ik(
                trans=cur_trans,
                rot=cur_rot,
                init_qpos=cur_qpos,
                retry_times=3
            )
            if success:
                valid_approach_points.insert(0, cur_qpos.copy())  # 插入到前面
            else:
                break
        except Exception:
            break
    
    # 从当前位置连接到第一个接近点
    traj_reach = []
    if len(valid_approach_points) > 1:
        # 从当前位置到第一个接近点
        first_approach_point = valid_approach_points[0]
        connection_steps = 20
        for i in range(connection_steps):
            alpha = (i + 1) / connection_steps
            interpolated_qpos = current_qpos + alpha * (first_approach_point - current_qpos)
            traj_reach.append(interpolated_qpos.copy())
        
        # 添加所有接近点
        traj_reach.extend(valid_approach_points)
    else:
        # 如果接近轨迹规划失败，直接从当前位置到抓取位置
        direct_steps = max(reach_steps, 20)
        for i in range(direct_steps):
            alpha = (i + 1) / direct_steps
            interpolated_qpos = current_qpos + alpha * (grasp_arm_qpos - current_qpos)
            traj_reach.append(interpolated_qpos.copy())
    
    # 阶段2: 保持位置，夹
    traj_squeeze = []
    for i in range(squeeze_steps):
        traj_squeeze.append(grasp_arm_qpos.copy())
    
    # 阶段3: 抬起轨迹
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
            # 如果终点IK成功，使用关节空间插值
            print(f"Lift IK successful, planning {lift_steps} step trajectory")
            for i in range(lift_steps):
                alpha = (i + 1) / lift_steps
                interpolated_qpos = grasp_arm_qpos + alpha * (lift_end_qpos - grasp_arm_qpos)
                traj_lift.append(interpolated_qpos.copy())
        else:
            # 如果终点IK失败，使用笛卡尔空间逐步规划
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
                        cur_qpos = new_qpos.copy()
                        traj_lift.append(cur_qpos.copy())
                    else:
                        # 如果某一步IK失败，使用关节空间插值继续
                        if len(traj_lift) > 0:
                            # 基于上一个成功的位置继续插值
                            prev_qpos = traj_lift[-1]
                            # 简单的向上插值（假设主要是第6个关节负责抬起）
                            lift_qpos = prev_qpos.copy()
                            lift_qpos[5] += 0.02  # 调整肩膀pitch关节
                            traj_lift.append(lift_qpos)
                        else:
                            # 如果没有成功的点，基于抓取位置简单插值
                            lift_qpos = grasp_arm_qpos.copy()
                            lift_qpos[5] += (step + 1) * 0.02
                            traj_lift.append(lift_qpos)
                except Exception:
                    # 异常情况下的备用方案
                    if len(traj_lift) > 0:
                        traj_lift.append(traj_lift[-1].copy())
                    else:
                        lift_qpos = grasp_arm_qpos.copy()
                        lift_qpos[5] += (step + 1) * 0.02
                        traj_lift.append(lift_qpos)
                        
    except Exception as e:
        print(f"Lift planning exception: {e}")
        print("Using fallback lift strategy - simple upward movement")
        
        # 备用方案：在笛卡尔空间中简单向上移动
        cur_trans = adjusted_grasp_trans.copy()
        cur_qpos = grasp_arm_qpos.copy()

        fallback_lift_height = 0.10  
        step_height = fallback_lift_height / lift_steps
        
        for step in range(lift_steps):
            cur_trans[2] += step_height
            
            # 尝试IK求解，如果失败就使用线性插值
            try:
                success, new_qpos = env.humanoid_robot_model.ik(
                    trans=cur_trans,
                    rot=target_rot,
                    init_qpos=cur_qpos,
                    retry_times=1  
                )
                if success:
                    cur_qpos = new_qpos.copy()
                    traj_lift.append(cur_qpos.copy())
                else:
                    # IK失败时，使用从当前位置的线性插值
                    if len(traj_lift) > 0:
                        # 基于上一个成功位置进行小幅插值
                        prev_qpos = traj_lift[-1]
                        # 模拟缓慢抬起
                        delta_qpos = (prev_qpos - grasp_arm_qpos) * 0.1  
                        next_qpos = prev_qpos + delta_qpos
                        traj_lift.append(next_qpos.copy())
                    else:
                        # 如果还没有成功点，就重复抓取位置（保持静止）
                        traj_lift.append(grasp_arm_qpos.copy())
            except Exception:
                # 静止不动
                if len(traj_lift) > 0:
                    traj_lift.append(traj_lift[-1].copy())
                else:
                    traj_lift.append(grasp_arm_qpos.copy())
    
    # 如果轨迹太短，补充到指定长度
    while len(traj_lift) < lift_steps:
        if len(traj_lift) > 0:
            traj_lift.append(traj_lift[-1].copy())
        else:
            traj_lift.append(grasp_arm_qpos.copy())
    
    print(f"Planned trajectories - Reach: {len(traj_reach)}, Squeeze: {len(traj_squeeze)}, Lift: {len(traj_lift)}")
    
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
    
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0,0,0,0,0]) # you can customize observing qpos to get wrist obs
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
        driller_pose = detect_driller_pose(rgb, depth,
                                            wrist_camera_matrix,
                                            camera_pose,
                                            args.est_drill_ckpt,
                                            args.device)
        # driller_pose = env.get_driller_pose()
        cprint(driller_pose, 'green')
        env.sim.debug_vis_pose(driller_pose, mocap_id='debug_axis_2') 

        set_trace()
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
            reach_steps=15,      # 接近步数
            lift_steps=12,       # 抬起步数
            squeeze_steps=8,     # 夹取步数
            delta_dist=0.03,     # 每步移动距离（较小，更精确）
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