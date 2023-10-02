/**
** Created by Zhijian QIAO.
** UAV Group, Hong Kong University of Science and Technology
** email: zqiaoac@connect.ust.hk
**/

#include "icp.h"
#include <pcl/registration/icp.h>
#include "parameters.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/time.h>
// #include <pcl/search/impl/kdtree.hpp>
// #include <pcl/kdtree/impl/kdtree_flann.hpp>
#include </usr/include/pcl-1.10/pcl/kdtree/kdtree_flann.h>
#include </usr/include/pcl-1.10/pcl/search/kdtree.h>


#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <typeinfo>



Eigen::Matrix4d icp_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guess) {
    // This is an example of using pcl::IterativeClosestPoint to align two point clouds
    // In your project, you should implement your own ICP algorithm!!!
    // In your implementation, you can use KDTree in PCL library to find nearest neighbors
    // Use chatGPT, google and github to learn how to use PCL library and implement ICP. But do not copy totally. TA will check your code using advanced tools.
    // If you use other's code, you should add a reference in your report. https://registry.hkust.edu.hk/resource-library/academic-integrity

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(src_cloud);
    icp.setInputTarget(tar_cloud);
    icp.setMaximumIterations(params::max_iterations);  // set maximum iteration
    icp.setTransformationEpsilon(1e-6);  // set transformation epsilon
    icp.setMaxCorrespondenceDistance(params::max_distance);  // set maximum correspondence distance
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp.align(aligned_cloud, init_guess.cast<float>());

    Eigen::Matrix4d transformation = icp.getFinalTransformation().cast<double>();
    return transformation;
}


/*  
    step 1: find the closest point from the corresponding point cloud
    step 2: formulate the least square problem and calculate R (rotation matrix)
    step 3: apply the R to the point cloud, and repeat step 1 to 3 until converge
*/
 
Eigen::Matrix4d my_icp1(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud, Eigen::Matrix4d init_guessR){
    // Downsampling
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tar_samp_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr samp_cloud;
    samp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    tar_samp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    voxel_grid.setInputCloud(src_cloud);
    voxel_grid.setLeafSize(0.1,0.1,0.1);
    voxel_grid.filter(*samp_cloud);
    voxel_grid.setInputCloud(tar_cloud);
    voxel_grid.setLeafSize(0.1,0.1,0.1);
    voxel_grid.filter(*tar_samp_cloud);
    std::cout << "down size *cloud_src_o from " << src_cloud->size() << " to " << samp_cloud->size() << std::endl;
	std::cout << "down size *cloud_tgt_o from " << tar_cloud->size() << " to " << tar_samp_cloud->points.size() << std::endl;


    pcl::PointCloud<pcl::PointXYZ>::Ptr tran_src_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tran_tar_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    Eigen::Matrix4d Rot;
    int iteration = 0;
    double error = 2.0;

	kdtree->setInputCloud(tar_samp_cloud);
    while(error > 0.00368683){
        std::cout << "iteration: " << iteration << std::endl;
        computeCorr(samp_cloud, tar_samp_cloud, tran_src_cloud, tran_tar_cloud, kdtree, error);
        std::cout << "Computed error: " << error << std::endl;
        // std::cout << "computeCorr src size: " << icp_src_cloud->points.size() << std::endl;
        std::cout << "computeCorr src size: " << samp_cloud->size() << std::endl;
        std::cout << "computeCorr tar size: " << tran_tar_cloud->points.size() << std::endl;
        Rot = estimateR(samp_cloud, tran_tar_cloud);
        pcl::transformPointCloud(*samp_cloud, *samp_cloud, Rot);
        std::cout << "THIS IS ONE POINTS CLOUD: " << samp_cloud->points[0].x << std::endl;

        iteration++;
    }
    print4x4Matrix(Rot);
    return Rot;
}




void computeCorr(pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr tar_cloud,
                pcl::PointCloud<pcl::PointXYZ>::Ptr& icp_src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& icp_tar_cloud, 
                pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree, double& error){
    icp_src_cloud->clear();
    icp_tar_cloud->clear();
    std::vector<int>indexs(src_cloud->size());
    double err = 0.0;

    for(int i=0; i<src_cloud->points.size(); i++){
        std::vector<int> idx(1);
        std::vector<float> sqrtDist(1);
        kdtree->nearestKSearch(src_cloud->points[i], 1, idx, sqrtDist);
        err = err + sqrtDist[0];
        indexs[i] = idx[0];
        
        // if(sqrtDist[0] < 0.02){}
        // icp_src_cloud->push_back(src_cloud->points[i]);
        // icp_tar_cloud->push_back(tar_cloud->points[idx[0]]);
    }
    pcl::copyPointCloud(*tar_cloud, indexs, *icp_tar_cloud);
    error = err/src_cloud->points.size();
}

void print4x4Matrix(const Eigen::Matrix4d matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.6f %6.6f %6.6f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.6f %6.6f %6.6f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.6f %6.6f %6.6f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.6f, %6.6f, %6.6f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

Eigen::Matrix4d estimateR(pcl::PointCloud<pcl::PointXYZ>::Ptr icp_src_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr icp_tar_cloud){
    int src_size = icp_src_cloud->points.size();
    int tar_size = icp_tar_cloud->points.size();
    if(src_size == tar_size){
        std::cout << "tar size: " << tar_size << std::endl;
        std::cout << "src size: " << src_size << std::endl;
        Eigen::Matrix<double, 3, Eigen::Dynamic> matrix_src(3, src_size), matrix_tar(3, tar_size);
        for (int i = 0; i < src_size; i++)
        {
            matrix_src(0, i) = icp_src_cloud->points[i].x;
            matrix_src(1, i) = icp_src_cloud->points[i].y;
            matrix_src(2, i) = icp_src_cloud->points[i].z;
            matrix_tar(0, i) = icp_tar_cloud->points[i].x;
            matrix_tar(1, i) = icp_tar_cloud->points[i].y;
            matrix_tar(2, i) = icp_tar_cloud->points[i].z;
        }
        Eigen::Matrix4d estmR = Eigen::umeyama(matrix_src, matrix_tar);
        // print4x4Matrix(estmR);
        
        return estmR;
    } else { 
        std::cout << "tar size != src size" << std::endl;
    }

}