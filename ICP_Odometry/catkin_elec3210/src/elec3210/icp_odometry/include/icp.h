/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#ifndef SRC_ICP_H
#define SRC_ICP_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h> 


Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess);
Eigen::Matrix4d my_icp1(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess);
void computeCorr(pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr,
                pcl::PointCloud<pcl::PointXYZ>::Ptr&, pcl::PointCloud<pcl::PointXYZ>::Ptr&, 
                pcl::search::KdTree<pcl::PointXYZ>::Ptr, double&);
Eigen::Matrix4d estimateR(pcl::PointCloud<pcl::PointXYZ>::Ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr);
void print4x4Matrix(const Eigen::Matrix4d);


#endif //SRC_ICP_H
