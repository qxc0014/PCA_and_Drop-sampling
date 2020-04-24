# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import math
import pandas as pd
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, israndom): 
    filtered_points = []
    # 作业3
    # 屏蔽开始
    #print(point_cloud) 
    label_x = point_cloud.loc[:,"x"]
    label_y = point_cloud.loc[:,"y"]
    label_z = point_cloud.loc[:,"z"]
    x_min = label_x.min()
    x_max = label_x.max()
    y_min = label_y.min()
    y_max = label_y.max()
    z_min = label_z.min()
    z_max = label_z.max()
    D_x = (x_max - x_min) / leaf_size
    D_y = (y_max - y_min) / leaf_size
    D_z = (z_max - z_min) / leaf_size
    print('X方向的voxel个数:',D_x,'Y方向的voxel个数:',D_y,'Z方向的voxel个数:',D_z)
    voxel_index = []
    #point_cloud.shape[0]
    for index in range(point_cloud.shape[0]):
        h_x = math.floor((point_cloud.iloc[index,0] - x_min)/leaf_size)
        h_y = math.floor((point_cloud.iloc[index,1] - y_min)/leaf_size)
        h_z = math.floor((point_cloud.iloc[index,2] - z_min)/leaf_size)
        voxelindex = math.floor(h_x + h_y*D_x + h_z * D_x * D_y)
        voxel_index.append(voxelindex)
    point_cloud['voxel'] =  voxel_index
    point_cloud_sort = point_cloud.sort_values(by='voxel',axis=0,ascending = True)
    '''True为随机选取，，False为均值'''
    if israndom == True:
        filtered_points = point_cloud_sort.drop_duplicates(['voxel'])
        filtered_points = filtered_points.drop(['voxel'],axis = 1,inplace = False)
    else: 
        grouped = point_cloud_sort.groupby('voxel')
        filtered_points = grouped['x','y','z'].agg(np.mean)
        print('重新分组',filtered_points)
    # 屏蔽结束

    print('降采样结果',filtered_points)
    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "/home/esoman/c++code/Homework I/ply/airplane/train/airplane_0001.ply"
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 20.0,False)#True为随机False为平均
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])

if __name__ == '__main__':
    main()
