#include "ekf_slam.h"
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/common/transforms.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace std;
using namespace Eigen;


EKFSLAM::~EKFSLAM() {}

EKFSLAM::EKFSLAM(ros::NodeHandle &nh):
        nh_(nh) {

//    initialize ros publisher
    lidar_sub = nh_.subscribe("/velodyne_points", 1, &EKFSLAM::cloudHandler, this);
    odom_sub = nh_.subscribe("/odom", 1, &EKFSLAM::odomHandler, this);
    map_cylinder_pub = nh_.advertise<visualization_msgs::MarkerArray>("/map_cylinder", 1);
    obs_cylinder_pub = nh_.advertise<visualization_msgs::MarkerArray>("/obs_cylinder", 1);
    odom_pub = nh_.advertise<nav_msgs::Odometry>("ekf_odom", 1000);
    path_pub = nh_.advertise<nav_msgs::Path>("ekf_path", 1000);
    scan_pub = nh_.advertise<sensor_msgs::PointCloud2>("current_scan", 1);
    map_pub = nh_.advertise<sensor_msgs::PointCloud2>("cloud_map", 1);
    laserCloudIn = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    mapCloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    extractCylinder = std::make_shared<ExtractCylinder>(nh_);

    globalId = -1;
	/**
	 * TODO: initialize the state vector and covariance matrix
	 */
    mState = Eigen::VectorXd::Zero(3); // x, y, yaw
    mCov = Eigen::MatrixXd::Identity(3,3);
    R = Eigen::MatrixXd::Identity(2,2);
    R << 1e-5, 0,
         0, 1e-1; // process noise
    Q = 1e+10*Eigen::MatrixXd::Identity(2,2); // measurement noise

    std::cout << "EKF SLAM initialized" << std::endl;
}

void EKFSLAM::run() {
    ros::Rate rate(1000);
    while (ros::ok()){
        if (cloudQueue.empty() || odomQueue.empty()){
            rate.sleep();
            continue;
        }

        cloudQueueMutex.lock();
        cloudHeader = cloudQueue.front().first;
        laserCloudIn = parseCloud(cloudQueue.front().second);
        cloudQueue.pop();
        cloudQueueMutex.unlock();

        // find the cloest odometry message
        odomMutex.lock();
        auto odomIter = odomQueue.front();
        auto odomPrevIter = odomQueue.front();
        while (!odomQueue.empty() && odomIter != odomQueue.back() && odomIter.first.stamp < cloudHeader.stamp){
            odomPrevIter = odomIter;
            odomIter = odomQueue.front();
            odomQueue.pop();
        }
        odomMutex.unlock();

        if (firstFrame){
            firstFrame = false;
            Twb = Eigen::Matrix4d::Identity();
            cloudHeaderLast = cloudHeader;
            continue;
        }

        auto odomMsg = odomIter == odomQueue.back() ? odomPrevIter : odomIter;
        Eigen::Vector2d ut = Eigen::Vector2d(odomMsg.second->twist.twist.linear.x, odomMsg.second->twist.twist.angular.z);
        double dt = (cloudHeader.stamp - cloudHeaderLast.stamp).toSec();

        timer.tic();
		// Extended Kalman Filter
		// 1. predict
        predictState(mState, mCov, ut, dt);
		// 2. update
        updateMeasurement();
		timer.toc();

		// publish odometry and map
		map_pub_timer.tic();
        accumulateMap();
        publishMsg();
		cloudHeaderLast = cloudHeader;

        rate.sleep();
    }
}

double EKFSLAM::normalizeAngle(double angle){
	if (angle > M_PI){
		angle -= 2 * M_PI;
	} else if (angle < -M_PI){
		angle += 2 * M_PI;
	}
	return angle;
}

Eigen::MatrixXd EKFSLAM::jacobGt(const Eigen::VectorXd& state, Eigen::Vector2d ut, double dt){
	int num_state = state.rows();
	Eigen::MatrixXd Gt = Eigen::MatrixXd::Identity(num_state, num_state);
	/**
	 * TODO: implement the Jacobian Gt
	 */
    Gt(0, 2) = -ut(0) * sin(state(2)) * dt;
    Gt(1, 2) = ut(0) * cos(state(2)) * dt;
	return Gt;
}

Eigen::MatrixXd EKFSLAM::jacobFt(const Eigen::VectorXd& state, Eigen::Vector2d ut, double dt){
	int num_state = state.rows();
	Eigen::MatrixXd Ft = Eigen::MatrixXd::Zero(num_state, 2);
	/**
	 * TODO: implement the Jacobian Ft
	 */
    Ft(0, 0) = dt * cos(state(2));
	Ft(1, 0) = dt * sin(state(2));
	Ft(2, 1) = dt;
	return Ft;
}

Eigen::MatrixXd EKFSLAM::jacobB(const Eigen::VectorXd& state, Eigen::Vector2d ut, double dt){
	int num_state = state.rows();
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(num_state, 2);
	B(0, 0) = dt * cos(state(2));
	B(1, 0) = dt * sin(state(2));
	B(2, 1) = dt;
	return B;
}

void EKFSLAM::predictState(Eigen::VectorXd& state, Eigen::MatrixXd& cov, Eigen::Vector2d ut, double dt){
    // Note: ut = [v, w]
	Eigen::MatrixXd Gt = jacobGt(state, ut, dt);
	Eigen::MatrixXd Ft = jacobFt(state, ut, dt);
    
    // std::cout << "Gt: \n" << Gt << std::endl;
    // std::cout << "Ft: \n" << Ft << std::endl;
    // std::cout << "state: \n" << state << std::endl;
    // std::cout << "cov: \n" << cov << std::endl;
    // std::cout << "state dim:" << state.size() << std::endl;
    // std::cout << "R dim:" << R.rows() << ", " << R.cols() << std::endl;
	state = state + jacobB(state, ut, dt) * ut; // update state
	cov = Gt * cov * Gt.transpose() + Ft * R * Ft.transpose(); // update covariance
    state(2) = normalizeAngle(state(2));
}

Eigen::Vector2d EKFSLAM::transform(const Eigen::Vector2d& p, const Eigen::Vector3d& x){
	Eigen::Vector2d p_t;
	p_t(0) = p(0) * cos(x(2)) - p(1) * sin(x(2)) + x(0);
	p_t(1) = p(0) * sin(x(2)) + p(1) * cos(x(2)) + x(1);
	return p_t;
}

void EKFSLAM::addNewLandmark(const Eigen::Vector2d& lm, const Eigen::MatrixXd& InitCov){
	// add new landmark to mState and mCov
	/**
	 * TODO: implement the function
	 */
    Eigen::VectorXd mState_copy = mState;
    mState.conservativeResize(mState.size()+2);
    mState.segment(0,mState_copy.size()) = mState_copy;
    mState(mState.size()-2) = lm(0);
    mState(mState.size()-1) = lm(1);
    
    int old_size = mCov.rows();
    int newlm_size = old_size + 2;
    Eigen::MatrixXd zeros_cov = Eigen::MatrixXd::Zero(2,2);
    mCov.conservativeResize(newlm_size, newlm_size);
    mCov.block(old_size, 0, 2,old_size) = Eigen::MatrixXd::Zero(2,old_size);
    mCov.block(0, old_size, old_size, 2) = Eigen::MatrixXd::Zero(old_size,2);
    mCov.block(old_size, old_size, 2, 2) = InitCov;

    
    // std::cout << "mCov dim:" << mCov.rows() << ", " << mCov.cols() << std::endl;
    // std::cout << "InitCov dim:" << InitCov.rows() << ", " << InitCov.cols() << std::endl;
    // std::cout << "addnewLandMark done" << std::endl;

}

void EKFSLAM::accumulateMap(){

    Eigen::Matrix4d Twb = Pose3DTo6D(mState.segment(0, 3));
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*laserCloudIn, *transformedCloud, Twb);
    *mapCloud += *transformedCloud;

    pcl::VoxelGrid<pcl::PointXYZ> voxelSampler;
    voxelSampler.setInputCloud(mapCloud);
    voxelSampler.setLeafSize(0.5, 0.5, 0.5);
    voxelSampler.filter(*mapCloud);
}

void EKFSLAM::updateMeasurement(){

    cylinderPoints = extractCylinder->extract(laserCloudIn, cloudHeader); // 2D pole centers in the laser/body frame
    Eigen::Vector3d xwb = mState.block<3, 1>(0, 0); // pose in the world frame
    int num_landmarks = (mState.rows() - 3) / 2; // number of landmarks in the state vector
    int num_obs = cylinderPoints.rows(); // number of observations
    Eigen::VectorXi indices = Eigen::VectorXi::Ones(num_obs) * -1; // indices of landmarks in the state vector
    Eigen::VectorXd shdist_hist = Eigen::VectorXd::Ones(num_obs) * -1; // record the shortest dist for each mState landmarks

    std::cout << "\nBIG loop: " << std::endl;
    for (int i = 0; i < num_obs; ++i) {
        Eigen::Vector2d pt_transformed = transform(cylinderPoints.row(i).transpose(), xwb); // 2D pole center in the world frame
		// Implement the data association here, i.e., find the corresponding landmark for each observation
		/**
		 * TODO: data association
		 * **/
        int min_index = -1;
        float min_dist = 6;
        std::cout << "\n SMALL loop: " << std::endl;
        for(int j = 0; j < num_landmarks; j++){
            float x = pt_transformed(0) - mState(3+(2*j));
            float y = pt_transformed(1) - mState(3+(2*j)+1);
            float dist = sqrt(pow(x,2)+pow(y,2));
            if(std::isnan(dist)){ continue; }
            if(min_dist > dist){
                min_index = j;
                min_dist = dist;
                indices(i) = min_index;
                shdist_hist(i) = min_dist;
                // std::cout << "DEBUG DONE" << std::endl;
            }




            /****** PRINT SECTION *********/ 
            std::cout << "j: " << j << std::endl;
            std::cout << "dist: " << dist << std::endl;
            std::cout << "min_dist: " << min_dist << std::endl;
            std::cout << "min_index: " << min_index << std::endl;
            
            // std::cout << "CHECK NAN" << std::endl;
            // std::cout << "j: " << j << std::endl;
            // std::cout << "pt_transformed: " << pt_transformed(0) << std::endl;
            // std::cout << "pt_transformed: " << pt_transformed(1) << std::endl;
            // std::cout << "mState(0): " << mState(3+(2*j)) << std::endl;
            // std::cout << "mState(1): " << mState(3+(2*j)+1) << std::endl;
            // std::cout << "x: " << x << std::endl;
            // std::cout << "y: " << y << std::endl;
            // std::cout << "dist: " << dist << std::endl;
            // std::cout << "min_dist: " << min_dist << std::endl;
            // std::cout << "min_index: " << min_index << std::endl;std::cout << "j: " << j << std::endl;
            // std::cout << "mState landmark: " << (mState.size()-3)/2 << std::endl;
            // std::cout << "num_obs: " << num_obs << std::endl;
            // std::cout << "globalId: " << globalId+1 << std::endl;
            /****** PRINT SECTION *********/ 
        }


        if (indices(i) == -1){
            indices(i) = ++globalId;
            addNewLandmark(pt_transformed, Q);
        }
        for(int ind_idx = 0; ind_idx < indices.size(); ind_idx++){
            int elementToCount = indices(ind_idx);
            int count = std::count(indices.data(), indices.data() + indices.size(), elementToCount);
            std::cout << "elementToCount: " << elementToCount << " = " << count << std::endl;
            if(count > 1 && elementToCount != -1){
                for(int ind_idx1 = 0; ind_idx1 < indices.size(); ind_idx1++){
                    if(indices(ind_idx1) == elementToCount){
                        if(shdist_hist(ind_idx1) > shdist_hist(ind_idx)){
                            indices(ind_idx) = elementToCount;
                            indices(ind_idx1) = -1;
                        } else {
                            indices(ind_idx) = -1;
                            indices(ind_idx1) = elementToCount;
                        }
                    }
                }
                std::cout << "LM MATCHING ERROR" << std::endl;
                std:cout << "count: " << count << std::endl;
                count = 0;
            }
            // while(count > 1)
        }
        std::cout << "mState landmark: " << (mState.size()-3)/2 << std::endl;
        std::cout << "num_obs: " << num_obs << std::endl;
    }
    // simulating bearing model
    Eigen::VectorXd z = Eigen::VectorXd::Zero(2 * num_obs);
    for (int i = 0; i < num_obs; ++i) {
        const Eigen::Vector2d& pt = cylinderPoints.row(i);
        z(2 * i, 0) = pt.norm();
        z(2 * i + 1, 0) = atan2(pt(1), pt(0));
    }
    // update the measurement vector
    num_landmarks = (mState.rows() - 3) / 2;
    for (int i = 0; i < num_obs; ++i) {
        int idx = indices(i);
        if (idx == -1 || idx + 1 > num_landmarks) continue;
        const Eigen::Vector2d& landmark = mState.block<2, 1>(3 + idx * 2, 0);
		// Implement the measurement update here, i.e., update the state vector and covariance matrix
		/**
		 * TODO: measurement update
         * Hi dim: (2N+3)*(2N+3)
		 */
        float dx = landmark(0) - xwb(0); 
        float dy = landmark(1) - xwb(1); 
        float q = pow(dx,2) + pow(dy,2);
        Eigen::Vector2d z_hat = Eigen::Vector2d::Zero(2);
        Eigen::Vector2d z_obs = Eigen::Vector2d::Zero(2);
        Eigen::MatrixXd Fi = Eigen::MatrixXd::Zero(5, mState.rows());
        Eigen::MatrixXd lHi = Eigen::MatrixXd::Zero(2, Fi.rows());
        lHi << -sqrt(q)*dx, -sqrt(q)*dy, 0, sqrt(q)*dx, sqrt(q)*dy,
                dy,         -dx,         -q, -dy,        dx;
        z_hat(0) = sqrt(q);
        z_hat(1) = atan2(dy,dx)-mState(2);
        z_obs(0) = z(2*i,0);
        z_obs(1) = z((2*i)+1,0);

        Fi.block(0,0,3,3) = Eigen::MatrixXd::Identity(3,3);
        int submatrix_idx = ((2*indices(i))+3);
        Fi.block(Fi.rows()-2, submatrix_idx, 2, 2) = Eigen::MatrixXd::Identity(2,2); 
        Eigen::MatrixXd Hi = (1/q) * lHi * Fi;

        Eigen::VectorXd mState_copy = mState;
        Eigen::MatrixXd mCov_copy   = mCov;
        Eigen::MatrixXd Kalman      = mCov * Hi.transpose() * ((Hi*mCov*Hi.transpose()+Q).inverse());
        Eigen::MatrixXd KH          = Kalman*Hi;
        Eigen::MatrixXd I           = Eigen::MatrixXd::Identity(KH.rows(), KH.cols());

        mState_copy = mState + Kalman*(z_obs - z_hat);
        mCov_copy = (I - KH)*mCov;

        mState = mState_copy;
        mCov = mCov_copy;
        // mState.segment(0,3)                    = mState_copy.segment(0,3);
        // mState((2*idx)+3)                      = mState_copy((2*idx)+3) ;
        // mState((2*idx)+3+1)                    = mState_copy((2*idx)+3+1);
        // mCov.block(0, 0, 3, 3)                 = mCov_copy.block(0, 0, 3, 3);
        // mCov.block((2*idx)+3, 0, 2, (2*idx)+3) = mCov_copy.block((2*idx)+3, 0, 2, (2*idx)+3);
        // mCov.block(0, (2*idx)+3, (2*idx)+3, 2) = mCov_copy.block(0, (2*idx)+3, (2*idx)+3, 2);
        // mCov.block((2*idx)+3, (2*idx)+3, 2, 2) = mCov_copy.block((2*idx)+3, (2*idx)+3, 2, 2);

        // std::cout << "Fi dim:" << Fi.rows() << ", " << Fi.cols() << std::endl;
        // std::cout << "mCov dim:" << mCov.rows() << ", " << mCov.cols() << std::endl;
        // std::cout << "Kalman dim:" << Kalman.rows() << ", " << Kalman.cols() << std::endl;
        // std::cout << "KH dim:" << KH.rows() << ", " << KH.cols() << std::endl;
        // std::cout << "I dim:" << I.rows() << ", " << I.cols() << std::endl;

    }
}

void EKFSLAM::publishMsg(){
    // publish map cylinder
    visualization_msgs::MarkerArray markerArray;
    int num_landmarks = (mState.rows() - 3) / 2;
    for (int i = 0; i < num_landmarks; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = cloudHeader.stamp;
        marker.ns = "map_cylinder";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = mState(3 + i * 2, 0);
        marker.pose.position.y = mState(3 + i * 2 + 1, 0);
        marker.pose.position.z = 0.5;
        marker.pose.orientation.x = 0;
        marker.pose.orientation.y = 0;
        marker.pose.orientation.z = 0;
        marker.pose.orientation.w = 1;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 1;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        markerArray.markers.push_back(marker);
    }
    map_cylinder_pub.publish(markerArray);

    int num_obs = cylinderPoints.rows();
    markerArray.markers.clear();
    for (int i = 0; i < num_obs; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = cloudHeader.stamp;
        marker.ns = "obs_cylinder";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;
        Eigen::Vector2d pt = transform(cylinderPoints.row(i).transpose(), mState.segment(0, 3));
        marker.pose.position.x = pt(0);
        marker.pose.position.y = pt(1);
        marker.pose.position.z = 0.5;
        marker.pose.orientation.x = 0;
        marker.pose.orientation.y = 0;
        marker.pose.orientation.z = 0;
        marker.pose.orientation.w = 1;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 1;
        marker.color.a = 1.0;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;

        markerArray.markers.push_back(marker);
    }
    obs_cylinder_pub.publish(markerArray);

//    publish odom
    nav_msgs::Odometry odom;
    odom.header.frame_id = "map";
    odom.child_frame_id = "base_link";
    odom.header.stamp = cloudHeader.stamp;
    odom.pose.pose.position.x = mState(0,0);
    odom.pose.pose.position.y = mState(1,0);
    odom.pose.pose.position.z = 0;
    Eigen::Quaterniond q(Eigen::AngleAxisd(mState(2,0), Eigen::Vector3d::UnitZ()));
    q.normalize();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom_pub.publish(odom);

//    publish path
    path.header.frame_id = "map";
    path.header.stamp = cloudHeader.stamp;
    geometry_msgs::PoseStamped pose;
    pose.header = odom.header;
    pose.pose = odom.pose.pose;
    path.poses.push_back(pose);
    path_pub.publish(path);

////    publish map
    sensor_msgs::PointCloud2 mapMsg;
    pcl::toROSMsg(*mapCloud, mapMsg);
    mapMsg.header.frame_id = "map";
    mapMsg.header.stamp = cloudHeader.stamp;
    map_pub.publish(mapMsg);

////    publish laser
    sensor_msgs::PointCloud2 laserMsg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserTransformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*laserCloudIn, *laserTransformed, Pose3DTo6D(mState.segment(0, 3)).cast<float>());
    pcl::toROSMsg(*laserTransformed, laserMsg);
    laserMsg.header.frame_id = "map";
    laserMsg.header.stamp = cloudHeader.stamp;
    scan_pub.publish(laserMsg);

	map_pub_timer.toc();
    std::cout << "x: " << mState(0,0) << " y: " << mState(1,0) << " theta: " << mState(2,0) * 180 / M_PI << ", time ekf: " << timer.duration_ms() << " ms"
						  << ", map_pub: " << map_pub_timer.duration_ms() << " ms" << std::endl;
}

void EKFSLAM::cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
    cloudQueueMutex.lock();
    std_msgs::Header cloudHeader = laserCloudMsg->header;
    cloudQueue.push(std::make_pair(cloudHeader, laserCloudMsg));
    cloudQueueMutex.unlock();
}

void EKFSLAM::odomHandler(const nav_msgs::OdometryConstPtr& odomMsg){
    odomMutex.lock();
    std_msgs::Header odomHeader = odomMsg->header;
    odomQueue.push(std::make_pair(odomHeader, odomMsg));
    odomMutex.unlock();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr EKFSLAM::parseCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudTmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*laserCloudMsg, *cloudTmp);
    // Remove Nan points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloudTmp, *cloudTmp, indices);
    return cloudTmp;
}

