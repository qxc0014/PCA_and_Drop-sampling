# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始    
    #计算中心
    center_data = np.mean(data)
    print(center_data)
    #去中心化
    new_data = data- center_data 
    #svd分解得到主成分
    U,sigma,VT = np.linalg.svd(np.transpose(new_data),False)
    eigenvectors_svd = U[:,[0]]
    #求协方差矩阵的特征向量为主成分
    if sort:
        covMat = np.cov(new_data, rowvar=0)
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))
        eigValIndice = np.argsort(eigVals)
        #print(eigValIndice)
        maxeigValIndice = eigValIndice[-1:-4:-1]
        eigenvectors = eigVects[:, maxeigValIndice]
        eigenvalues = eigVals[maxeigValIndice]
    # 屏蔽结束
    return eigenvalues, eigenvectors 

def main():
    # 指定点云路径
    root_dir = '/media/esoman/Data1/数据集/ply/' # 数据集路径
    cat = os.listdir(root_dir)
    for i in range(len(cat)):
        filename = os.path.join(root_dir, cat[i],'train', cat[i]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云
    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector_x = v[:,0] #点云主方向对应的向量
    point_cloud_vector_y = v[:,1] #点云次方向对应的向量
    # TODO: 此处只显示了点云，还没有显示PCA
    x = np.array(point_cloud_vector_x)
    y = np.array(point_cloud_vector_y)
    x = pd.Series(x[:,0])
    y = pd.Series(y[:,0])
    x_coor = points*x.rename(index = {0:'x',1:'y',2:'z'})
    y_coor = points*y.rename(index = {0:'x',1:'y',2:'z'})
    x_dir = x_coor['x'] + x_coor['y'] +x_coor['z'] 
    y_dir = y_coor['x'] + y_coor['y'] +y_coor['z'] 
    #new_points = x_dir * x np.array([x_dir]).T
    new_points_x = np.array([x_dir]).T*np.array(point_cloud_vector_x).T
    new_points_y = np.array([y_dir]).T*np.array(point_cloud_vector_y).T
    new_points = new_points_x + new_points_y
    new_points = pd.DataFrame(new_points,columns=['x','y','z'])
    platform_pcd = o3d.geometry.PointCloud()
    platform_pcd.points = o3d.utility.Vector3dVector(np.array(new_points))
    print('new_points',new_points)
    b = np.array([0,0,0])
    vectorpoints = np.insert(v.T*100,0,values=b,axis=0)#
    print(vectorpoints)
    lines = [
        [0, 1],
        [0, 2],
    ]
    line_pcd = o3d.geometry.LineSet(points= o3d.utility.Vector3dVector(vectorpoints),
        lines=o3d.utility.Vector2iVector(lines),)#用于显示主成分轴与次主成分轴
    #o3d.visualization.draw_geometries([point_cloud_o3d]+[line_pcd])
    #o3d.visualization.draw_geometries([line_pcd])
    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    #print(pcd_tree)
    for index in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[index], 6)
        neighbor_points = points.iloc[idx[0:]]#拿出每个点的领域点的index
        print('相邻点的集合:',neighbor_points)
        neighbor_w,neighbor_v = PCA(neighbor_points)#放入PCA中计算
        neighbor = neighbor_v[:,2]#得到法线
        if index == 0:
            normals = np.array(neighbor.T, dtype=np.float64)#用于初始化法线
        else:
            normals = np.append(normals,np.array(neighbor.T, dtype=np.float64),axis=0)
    print('法线方向',normals)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([platform_pcd])


if __name__ == '__main__':
    main()
