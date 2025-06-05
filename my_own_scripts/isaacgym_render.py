import numpy as np
import isaacgym
from isaacgym import gymapi
import cv2

def get_scene_images(
    camera_pos: np.ndarray,
    look_at: np.ndarray,
    table_urdf: str,
    table_position: np.ndarray,
    drill_urdf: str,
    drill_position: np.ndarray,
    quad_urdf: str,
    quad_position: np.ndarray,
    quad_joint_values: np.ndarray,
    img_width: int = 640,
    img_height: int = 480,
) -> dict:
    """
    根据给定的相机位置、桌子、钻头和四足机械狗的URDF及位置，生成RGB和深度图像。

    参数:
        camera_pos (np.ndarray): 相机位置，形状为(3,)的numpy数组
        look_at (np.ndarray): 相机朝向点，形状为(3,)的numpy数组
        table_urdf (str): 桌子URDF文件的相对路径
        table_position (np.ndarray): 桌子位置，形状为(3,)的numpy数组
        drill_urdf (str): 钻头URDF文件的相对路径
        drill_position (np.ndarray): 钻头位置，形状为(3,)的numpy数组
        quad_urdf (str): 四足机械狗URDF文件的相对路径
        quad_position (np.ndarray): 四足机械狗位置，形状为(3,)的numpy数组
        quad_joint_values (np.ndarray): 四足机械狗关节值，形状为(num_dofs,)的numpy数组
        img_width (int): 输出图像宽度，默认为640
        img_height (int): 输出图像高度，默认为480

    返回:
        dict: 包含RGB图像和深度图的字典，x["rgb"]为形状(height, width, 3)的RGB图像，
              x["depth"]为形状(height, width)的深度图（单位：米）
    """
    # 初始化Isaac Gym
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # 创建环境
    lower = gymapi.Vec3(-2, -2, 0)
    upper = gymapi.Vec3(2, 2, 2)
    env = gym.create_env(sim, lower, upper, 1)

    # 加载资产（参考combined_sim.py中的加载方式）
    table_asset_options = gymapi.AssetOptions()
    table_asset_options.fix_base_link = True  # 桌子固定在环境中
    table_asset = gym.load_asset(sim, asset_root, table_urdf, table_asset_options)

    drill_asset_options = gymapi.AssetOptions()
    drill_asset_options.fix_base_link = True  # 钻头固定在桌子上
    drill_asset = gym.load_asset(sim, asset_root, drill_urdf, drill_asset_options)

    quad_asset_options = gymapi.AssetOptions()
    quad_asset = gym.load_asset(sim, asset_root, quad_urdf, quad_asset_options)

    # 创建演员并设置位置
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(*table_position)
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

    drill_pose = gymapi.Transform()
    drill_pose.p = gymapi.Vec3(*drill_position)
    drill_handle = gym.create_actor(env, drill_asset, drill_pose, "drill", 0, 0)

    quad_pose = gymapi.Transform()
    quad_pose.p = gymapi.Vec3(*quad_position)
    quad_handle = gym.create_actor(env, quad_asset, quad_pose, "quad", 0, 0)

    # 设置四足机械狗的关节值
    num_dofs = gym.get_asset_dof_count(quad_asset)
    assert len(quad_joint_values) == num_dofs, f"关节值数量（{len(quad_joint_values)}）与自由度数量（{num_dofs}）不匹配"
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    for i in range(num_dofs):
        dof_states['pos'][i] = quad_joint_values[i]
    gym.set_actor_dof_states(env, quad_handle, dof_states, gymapi.STATE_POS)

    # 设置相机
    camera_props = gymapi.CameraProperties()
    camera_props.width = img_width
    camera_props.height = img_height
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(*camera_pos), gymapi.Vec3(*look_at))

    # 模拟一步以更新场景
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    # 渲染相机传感器
    gym.render_all_camera_sensors(sim)

    # 获取图像
    rgb_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_COLOR)
    depth_image = gym.get_camera_image(sim, env, camera_handle, gymapi.IMAGE_DEPTH)

    # 处理图像数据
    rgb = rgb_image.reshape(img_height, img_width, 4)[:, :, :3]  # 从RGBA转换为RGB
    depth = depth_image.reshape(img_height, img_width)  # 深度图，单位为米

    # 清理资源
    gym.destroy_env(env)
    gym.destroy_sim(sim)

    return {"rgb": rgb, "depth": depth}

if __name__ == "__main__":
    # 示例用法
    camera_pos = np.array([1.0, 1.0, 1.0])  # 相机位置
    look_at = np.array([0.0, 0.0, 0.0])     # 相机朝向点
    table_position = np.array([0.0, 0.0, 0.0])  # 桌子位置
    drill_position = np.array([0.0, 0.0, 0.5])  # 钻头位置（在桌子上）
    quad_position = np.array([0.5, 0.5, 0.0])   # 四足机械狗位置
    quad_joint_values = np.zeros(12)  # 假设四足机械狗有12个关节
    asset_root = "path/to/assets"     # 资产根目录（需替换为实际路径）
    table_urdf = "table.urdf"         # 桌子URDF文件路径
    drill_urdf = "drill.urdf"         # 钻头URDF文件路径
    quad_urdf = "quad.urdf"           # 四足机械狗URDF文件路径

    # 调用函数生成图像
    x = get_scene_images(
        camera_pos=camera_pos,
        look_at=look_at,
        table_urdf=table_urdf,
        table_position=table_position,
        drill_urdf=drill_urdf,
        drill_position=drill_position,
        quad_urdf=quad_urdf,
        quad_position=quad_position,
        quad_joint_values=quad_joint_values,
        asset_root=asset_root,
    )

    # 保存或显示图像
    cv2.imwrite("rgb.png", x["rgb"])
    cv2.imwrite("depth.png", (x["depth"] * 1000).astype(np.uint16))  # 深度图保存为毫米单位