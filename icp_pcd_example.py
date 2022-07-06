# examples/Python/Basic/icp_registration.py

import open3d as o3d
import numpy as np
import laspy as lp
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("/home/uros/Documents/Projects/ouster_lidar/las_moving/las_out_000000.xyz")
    target = o3d.io.read_point_cloud("/home/uros/Documents/Projects/ouster_lidar/las_moving/las_out_000001.xyz")
    #input_path = "/home/uros/Documents/Projects/ouster_lidar/las_moving/las_out_000000.las"
    #point_cloud1 = lp.read(input_path)
    #points1 = np.vstack((point_cloud1.x, point_cloud1.y, point_cloud1.z)).transpose()
    #colors1 = np.vstack((point_cloud1.red, point_cloud1.green, point_cloud1.blue)).transpose()
    
    #pcd1 = o3d.geometry.PointCloud()
    #pcd1.points = o3d.utility.Vector3dVector(points1)
    #pcd1.colors = o3d.utility.Vector3dVector(colors1/65535)
    #pcd1.normals = o3d.utility.Vector3dVector(normals)
    
    #input_path = "/home/uros/Documents/Projects/ouster_lidar/las_moving/las_out_000001.las"
    #point_cloud2 = lp.read(input_path)
    #points2 = np.vstack((point_cloud2.x, point_cloud2.y, point_cloud2.z)).transpose()
    #colors2 = np.vstack((point_cloud2.red, point_cloud2.green, point_cloud2.blue)).transpose()
    
    #pcd2 = o3d.geometry.PointCloud()
    #pcd2.points = o3d.utility.Vector3dVector(points2)
    #pcd2.colors = o3d.utility.Vector3dVector(colors2/65535)
    #pcd1.normals = o3d.utility.Vector3dVector(normals)
    
    #source = pcd1
    #target = pcd2

    threshold = 0.02
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0], 
                             [0, 0, 0, 0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
