import os, numpy as np, trimesh as tm, torch, open3d as o3d

from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes


def get_obj_pcd(obj, num_points=2048):
    """
    从给定的三维模型中获取点云数据。

    参数:
        obj (trimesh.Trimesh): 三维模型对象
        num_points (int): 需要采样的点数

    返回:
        np.ndarray: 形状为(num_points, 3)的点云数据
    """
    # 将三角网格转换为PyTorch3D的Meshes格式
    verts = torch.tensor(obj.vertices, dtype=torch.float32).unsqueeze(0)
    faces = torch.tensor(obj.faces, dtype=torch.int64).unsqueeze(0)
    meshes = Meshes(verts=verts, faces=faces)

    # 从网格中采样点云
    pcd = sample_points_from_meshes(meshes, num_samples=num_points*32)
    pcd = sample_farthest_points(pcd, K=num_points)[0][0].cpu().numpy()

    return pcd

if __name__ == "__main__":
    obj = tm.load('asset/obj/power_drill/single.obj')
    pcd = get_obj_pcd(obj, num_points=2048)
    # 用o3d可视化
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    obj_o3d = o3d.io.read_triangle_mesh('asset/obj/power_drill/single.obj')
    obj_o3d.compute_vertex_normals()
    o3d.visualization.draw_geometries([pcd_o3d, obj_o3d])
    # 保存点云数据
    np.save('asset/obj/power_drill/single_pcd.npy', pcd)
