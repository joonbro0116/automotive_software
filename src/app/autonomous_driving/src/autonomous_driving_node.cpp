/**
 * @copyright Hanyang University, Department of Automotive Engineering, 2024. All rights reserved.
 *            Subject to limited distribution and restricted disclosure only.
 *
 * @file      autonomous_driving_node.cpp
 * @brief     autonomous driving node
 *
 * @date      2018-11-20 created by Kichun Jo (kichunjo@hanyang.ac.kr)
 *            2023-08-07 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : adapt new template
 *            2023-08-20 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : change to ROS2
 *            2024-11-05 updated by Yuseung Na (yuseungna@hanyang.ac.kr)
 *              : clean up
 */

#include "autonomous_driving_node.hpp"
#include <algorithm>

AutonomousDriving::AutonomousDriving(const std::string &node_name, const rclcpp::NodeOptions &options)
    : Node(node_name, options) {

    // QoS init
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));

    // Parameters
    this->declare_parameter("autonomous_driving/ns", "");
    this->declare_parameter("autonomous_driving/loop_rate_hz", 100.0);
    this->declare_parameter("autonomous_driving/use_manual_inputs", false);    
    ////////////////////// TODO //////////////////////
    // TODO: Add more parameters

    //////////////////////////////////////////////////
    ProcessParams();

    RCLCPP_INFO(this->get_logger(), "vehicle_namespace: %s", cfg_.vehicle_namespace.c_str());
    RCLCPP_INFO(this->get_logger(), "loop_rate_hz: %f", cfg_.loop_rate_hz);
    RCLCPP_INFO(this->get_logger(), "use_manual_inputs: %d", cfg_.use_manual_inputs);
    ////////////////////// TODO //////////////////////
    // TODO: Add more parameters

    //////////////////////////////////////////////////

    // Subscriber init
    s_manual_input_ = this->create_subscription<ad_msgs::msg::VehicleCommand>(
        "/manual_input", qos_profile, std::bind(&AutonomousDriving::CallbackManualInput, this, std::placeholders::_1));
    s_vehicle_state_ = this->create_subscription<ad_msgs::msg::VehicleState>(
        "vehicle_state", qos_profile, std::bind(&AutonomousDriving::CallbackVehicleState, this, std::placeholders::_1));
    s_lane_points_ = this->create_subscription<ad_msgs::msg::LanePointData>(
        "lane_points", qos_profile, std::bind(&AutonomousDriving::CallbackLanePoints, this, std::placeholders::_1));
    s_mission_ = this->create_subscription<ad_msgs::msg::Mission>(
        "mission", qos_profile, std::bind(&AutonomousDriving::CallbackMission, this, std::placeholders::_1));

    // Publisher init
    p_vehicle_command_ = this->create_publisher<ad_msgs::msg::VehicleCommand>(
        "vehicle_command", qos_profile);
    p_driving_way_ = this->create_publisher<ad_msgs::msg::PolyfitLaneData>(
        "driving_way", qos_profile);
    p_poly_lanes_ = this->create_publisher<ad_msgs::msg::PolyfitLaneDataArray>(
        "poly_lanes", qos_profile);
    p_lane_sample_points_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "lane_sample_points", qos_profile);
    p_points_memory_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "points_memory", qos_profile);
    p_polyfit_lanes_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "polyfit_lanes_marker", qos_profile);
    p_driving_way_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "driving_way_marker", qos_profile);
    p_lane_info_text_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "lane_info_text", qos_profile);

    // Timer init
    t_run_node_ = this->create_wall_timer(
        std::chrono::milliseconds((int64_t)(1000 / cfg_.loop_rate_hz)),
        [this]() { this->Run(); }); 
}

AutonomousDriving::~AutonomousDriving() {}

void AutonomousDriving::ProcessParams() {
    this->get_parameter("autonomous_driving/ns", cfg_.vehicle_namespace);
    this->get_parameter("autonomous_driving/loop_rate_hz", cfg_.loop_rate_hz);
    this->get_parameter("autonomous_driving/use_manual_inputs", cfg_.use_manual_inputs);
    ////////////////////// TODO //////////////////////
    // TODO: Add more parameters

    //////////////////////////////////////////////////
}

void AutonomousDriving::Run() {
    auto current_time = this->now();
    RCLCPP_INFO_THROTTLE(this->get_logger(), *get_clock(), 1000, "Running ...");
    ProcessParams();

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Get subscribe variables
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    if (cfg_.use_manual_inputs == true) {
        if (b_is_manual_input_ == false) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Manual Input ...");
            return;
        }
    }

    if (b_is_simulator_on_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Vehicle State ...");
        return;
    }

    if (b_is_lane_points_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Lane Points ...");
        return;
    }

    if (b_is_mission_ == false) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Wait for Mission ...");
        return;
    }

    interface::VehicleCommand manual_input; {
        if (cfg_.use_manual_inputs == true) {
            std::lock_guard<std::mutex> lock(mutex_manual_input_);
            manual_input = i_manual_input_;
        }
    }

    interface::VehicleState vehicle_state; {
        std::lock_guard<std::mutex> lock(mutex_vehicle_state_);
        vehicle_state = i_vehicle_state_;
    }

    interface::Lane lane_points; {
        std::lock_guard<std::mutex> lock(mutex_lane_points_);
        lane_points = i_lane_points_;
    }

    interface::Mission mission; {
        std::lock_guard<std::mutex> lock(mutex_mission_);
        mission = i_mission_;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Algorithm
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

    // Lane processing and ego driveway estimation
    interface::PolyfitLanes poly_lanes = lane_processor_.Process(lane_points, vehicle_state);
    interface::PolyfitLane driving_way = lane_processor_.GetEgoLane();
    // frame_id 항상 설정 (ego_lane_에서 설정 안 될 경우 대비)
    if (driving_way.frame_id.empty()) {
        driving_way.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
    }

    interface::VehicleCommand vehicle_command;

    if (cfg_.use_manual_inputs == true) {
        vehicle_command = manual_input;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    // Publish output
    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
    p_vehicle_command_->publish(ros2_bridge::UpdateVehicleCommand(vehicle_command));
    p_driving_way_->publish(ros2_bridge::UpdatePolyfitLane(driving_way));
    p_poly_lanes_->publish(ros2_bridge::UpdatePolyfitLanes(poly_lanes));

    // ===== PolyLane LINE + POINTS 시각화 =====
    visualization_msgs::msg::MarkerArray marker_array;
    int marker_id = 0;

    // Lane별 색상 정의
    std::map<std::string, std::array<double, 3>> lane_colors{
        {"lane1", {1.0, 0.0, 1.0}},        // Magenta (외분 좌측)
        {"lane2", {1.0, 0.0, 0.0}},        // Red (관찰 좌측)
        {"lane3", {0.0, 0.0, 1.0}},        // Blue (관찰 우측)
        {"lane4", {1.0, 1.0, 0.0}},        // Yellow (외분 우측)
        {"ego_center", {0.0, 1.0, 0.0}}    // Green (중심선)
    };

    // LINE_STRIP 마커 생성 (폴리핏 레인)
    for (const auto& poly_lane : poly_lanes.polyfitlanes) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
        marker.header.stamp = current_time;
        marker.ns = "polylanes";
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.scale.x = 0.15;  // Line width
        marker.lifetime = rclcpp::Duration::from_seconds(0.5);

        // 색상 설정
        auto it = lane_colors.find(poly_lane.id);
        if (it != lane_colors.end()) {
            marker.color.r = it->second[0];
            marker.color.g = it->second[1];
            marker.color.b = it->second[2];
        } else {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
        }
        marker.color.a = 0.9;

        // 다항식 샘플링
        const double x_start = -15.0;
        const double x_end = 20.0;
        const double step = 0.5;

        for (double x = x_start; x <= x_end; x += step) {
            double y = poly_lane.a0 + poly_lane.a1 * x
                     + poly_lane.a2 * x * x
                     + poly_lane.a3 * x * x * x;

            geometry_msgs::msg::Point p;
            p.x = x;
            p.y = y;
            p.z = 0.05;
            marker.points.push_back(p);
        }

        marker_array.markers.push_back(marker);
    }

    // SPHERE_LIST 마커 생성 (샘플 포인트)
    // lane_processor 내부와 동일한 범위/개수로 샘플 포인트 시각화
    // PRIMARY_LANE_NUM_SAMPLES = 80, MEMORY_X_MIN = -15.0, MEMORY_X_MAX = 20.0

    for (const auto& poly_lane : poly_lanes.polyfitlanes) {
        // polyfit의 실제 x 범위 사용 (내부 피팅 범위와 동일)
        double sample_x_start = poly_lane.x_start;
        double sample_x_end = poly_lane.x_end;

        // 유효하지 않은 범위인 경우 기본값 사용
        const bool invalid_x_range = (sample_x_end <= sample_x_start);
        if (invalid_x_range) {
            sample_x_start = -15.0;  // MEMORY_X_MIN
            sample_x_end = 20.0;     // MEMORY_X_MAX
        }

        const int num_points = 80;  // PRIMARY_LANE_NUM_SAMPLES와 동일
        const double x_step = (sample_x_end - sample_x_start) / (num_points - 1);

        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
        marker.header.stamp = current_time;
        marker.ns = "lane_sample_points_" + poly_lane.id;
        marker.id = marker_id++;
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2;  // Sphere diameter (smaller for more points)
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        marker.lifetime = rclcpp::Duration::from_seconds(0.5);  // Marker lifetime

        // 색상 설정
        auto it = lane_colors.find(poly_lane.id);
        if (it != lane_colors.end()) {
            marker.color.r = it->second[0];
            marker.color.g = it->second[1];
            marker.color.b = it->second[2];
        } else {
            marker.color.r = 1.0;
            marker.color.g = 1.0;
            marker.color.b = 1.0;
        }
        marker.color.a = 0.5;

        for (int i = 0; i < num_points; ++i) {
            double x_vehicle = sample_x_start + i * x_step;
            double y_vehicle = poly_lane.a0 + poly_lane.a1 * x_vehicle
                             + poly_lane.a2 * x_vehicle * x_vehicle
                             + poly_lane.a3 * x_vehicle * x_vehicle * x_vehicle;

            // NaN/Inf 체크
            if (!std::isfinite(y_vehicle)) {
                continue;
            }

            geometry_msgs::msg::Point p;
            p.x = x_vehicle;
            p.y = y_vehicle;
            p.z = 0.1;  // Slightly above ground

            marker.points.push_back(p);
        }

        marker_array.markers.push_back(marker);
    }

    p_lane_sample_points_->publish(marker_array);

    // ===== Lane Info TEXT 마커 (별도 토픽) =====
    visualization_msgs::msg::MarkerArray text_marker_array;
    int text_marker_id = 0;

    // Driveway 정보 TEXT 마커
    visualization_msgs::msg::Marker driveway_text;
    driveway_text.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
    driveway_text.header.stamp = current_time;
    driveway_text.ns = "driveway_info";
    driveway_text.id = text_marker_id++;
    driveway_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    driveway_text.action = visualization_msgs::msg::Marker::ADD;

    driveway_text.pose.position.x = 5.0;
    driveway_text.pose.position.y = 0.0;
    driveway_text.pose.position.z = 3.0;
    driveway_text.pose.orientation.w = 1.0;

    driveway_text.scale.z = 1.0;  // Text height
    driveway_text.color.r = 1.0;
    driveway_text.color.g = 1.0;
    driveway_text.color.b = 0.0;
    driveway_text.color.a = 1.0;

    int current_driveway = lane_processor_.GetCurrentDriveway();
    driveway_text.text = "Current Driveway: " + std::to_string(current_driveway);
    driveway_text.lifetime = rclcpp::Duration::from_seconds(0.5);

    text_marker_array.markers.push_back(driveway_text);

    // 유효한 클러스터 정보 TEXT 마커
    visualization_msgs::msg::Marker cluster_text;
    cluster_text.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
    cluster_text.header.stamp = current_time;
    cluster_text.ns = "cluster_info";
    cluster_text.id = text_marker_id++;
    cluster_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    cluster_text.action = visualization_msgs::msg::Marker::ADD;

    cluster_text.pose.position.x = 5.0;
    cluster_text.pose.position.y = 0.0;
    cluster_text.pose.position.z = 4.5;
    cluster_text.pose.orientation.w = 1.0;

    cluster_text.scale.z = 0.8;
    cluster_text.color.r = 0.0;
    cluster_text.color.g = 1.0;
    cluster_text.color.b = 1.0;
    cluster_text.color.a = 1.0;

    int valid_count = lane_processor_.GetValidClusterCount();
    std::string lane1_status = lane_processor_.IsLane1Valid() ? "O" : "X";
    std::string lane2_status = lane_processor_.IsLane2Valid() ? "O" : "X";
    std::string lane3_status = lane_processor_.IsLane3Valid() ? "O" : "X";
    std::string lane4_status = lane_processor_.IsLane4Valid() ? "O" : "X";

    cluster_text.text = "Valid Clusters: " + std::to_string(valid_count) + "/4\n"
                      + "L1:" + lane1_status + " L2:" + lane2_status
                      + " L3:" + lane3_status + " L4:" + lane4_status;
    cluster_text.lifetime = rclcpp::Duration::from_seconds(0.5);

    text_marker_array.markers.push_back(cluster_text);

    // 각 레인별 포인트 수 TEXT 마커
    visualization_msgs::msg::Marker points_count_text;
    points_count_text.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
    points_count_text.header.stamp = current_time;
    points_count_text.ns = "points_count_info";
    points_count_text.id = text_marker_id++;
    points_count_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    points_count_text.action = visualization_msgs::msg::Marker::ADD;

    points_count_text.pose.position.x = 5.0;
    points_count_text.pose.position.y = 0.0;
    points_count_text.pose.position.z = 5.5;
    points_count_text.pose.orientation.w = 1.0;

    points_count_text.scale.z = 0.6;
    points_count_text.color.r = 1.0;
    points_count_text.color.g = 1.0;
    points_count_text.color.b = 1.0;
    points_count_text.color.a = 1.0;

    size_t lane1_pts = lane_processor_.GetLane1Points().size();
    size_t lane2_pts = lane_processor_.GetLane2Points().size();
    size_t lane3_pts = lane_processor_.GetLane3Points().size();
    size_t lane4_pts = lane_processor_.GetLane4Points().size();

    points_count_text.text = "Points: L1=" + std::to_string(lane1_pts)
                           + " L2=" + std::to_string(lane2_pts)
                           + " L3=" + std::to_string(lane3_pts)
                           + " L4=" + std::to_string(lane4_pts);
    points_count_text.lifetime = rclcpp::Duration::from_seconds(0.5);

    text_marker_array.markers.push_back(points_count_text);

    p_lane_info_text_->publish(text_marker_array);

    // Visualize points memory in vehicle frame
    visualization_msgs::msg::MarkerArray points_memory_array;
    const std::string memory_frame = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;

    // Lane2 points memory (green)
    visualization_msgs::msg::Marker lane2_memory_marker;
    lane2_memory_marker.header.frame_id = memory_frame;
    lane2_memory_marker.header.stamp = current_time;
    lane2_memory_marker.ns = "points_memory_lane2";
    lane2_memory_marker.id = 0;
    lane2_memory_marker.type = visualization_msgs::msg::Marker::POINTS;
    lane2_memory_marker.action = visualization_msgs::msg::Marker::ADD;
    lane2_memory_marker.scale.x = 0.1;  // Point width
    lane2_memory_marker.scale.y = 0.1;  // Point height
    lane2_memory_marker.color.r = 0.0;
    lane2_memory_marker.color.g = 1.0;
    lane2_memory_marker.color.b = 0.0;
    lane2_memory_marker.color.a = 0.8;
    lane2_memory_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    for (const auto& pt : lane_processor_.getLane2PointsMemoryVehicleFrame(vehicle_state)) {
        lane2_memory_marker.points.push_back(pt);
    }
    points_memory_array.markers.push_back(lane2_memory_marker);

    // Lane3 points memory (blue)
    visualization_msgs::msg::Marker lane3_memory_marker;
    lane3_memory_marker.header.frame_id = memory_frame;
    lane3_memory_marker.header.stamp = current_time;
    lane3_memory_marker.ns = "points_memory_lane3";
    lane3_memory_marker.id = 1;
    lane3_memory_marker.type = visualization_msgs::msg::Marker::POINTS;
    lane3_memory_marker.action = visualization_msgs::msg::Marker::ADD;
    lane3_memory_marker.scale.x = 0.1;  // Point width
    lane3_memory_marker.scale.y = 0.1;  // Point height
    lane3_memory_marker.color.r = 0.0;
    lane3_memory_marker.color.g = 0.0;
    lane3_memory_marker.color.b = 1.0;
    lane3_memory_marker.color.a = 0.8;
    lane3_memory_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    for (const auto& pt : lane_processor_.getLane3PointsMemoryVehicleFrame(vehicle_state)) {
        lane3_memory_marker.points.push_back(pt);
    }
    points_memory_array.markers.push_back(lane3_memory_marker);

    // Lane1 points memory (red)
    visualization_msgs::msg::Marker lane1_memory_marker;
    lane1_memory_marker.header.frame_id = memory_frame;
    lane1_memory_marker.header.stamp = current_time;
    lane1_memory_marker.ns = "points_memory_lane1";
    lane1_memory_marker.id = 2;
    lane1_memory_marker.type = visualization_msgs::msg::Marker::POINTS;
    lane1_memory_marker.action = visualization_msgs::msg::Marker::ADD;
    lane1_memory_marker.scale.x = 0.1;
    lane1_memory_marker.scale.y = 0.1;
    lane1_memory_marker.color.r = 1.0;
    lane1_memory_marker.color.g = 0.0;
    lane1_memory_marker.color.b = 0.0;
    lane1_memory_marker.color.a = 0.8;
    lane1_memory_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    for (const auto& pt : lane_processor_.getLane1PointsMemoryVehicleFrame(vehicle_state)) {
        lane1_memory_marker.points.push_back(pt);
    }
    points_memory_array.markers.push_back(lane1_memory_marker);

    // Lane4 points memory (yellow)
    visualization_msgs::msg::Marker lane4_memory_marker;
    lane4_memory_marker.header.frame_id = memory_frame;
    lane4_memory_marker.header.stamp = current_time;
    lane4_memory_marker.ns = "points_memory_lane4";
    lane4_memory_marker.id = 3;
    lane4_memory_marker.type = visualization_msgs::msg::Marker::POINTS;
    lane4_memory_marker.action = visualization_msgs::msg::Marker::ADD;
    lane4_memory_marker.scale.x = 0.1;
    lane4_memory_marker.scale.y = 0.1;
    lane4_memory_marker.color.r = 1.0;
    lane4_memory_marker.color.g = 1.0;
    lane4_memory_marker.color.b = 0.0;
    lane4_memory_marker.color.a = 0.8;
    lane4_memory_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    for (const auto& pt : lane_processor_.getLane4PointsMemoryVehicleFrame(vehicle_state)) {
        lane4_memory_marker.points.push_back(pt);
    }
    points_memory_array.markers.push_back(lane4_memory_marker);

    p_points_memory_->publish(points_memory_array);

    // ===== PolyfitLanes Marker 별도 publish (RViz 호환용) =====
    visualization_msgs::msg::MarkerArray polyfit_lanes_marker_array;
    int polyfit_marker_id = 0;

    for (const auto& poly_lane : poly_lanes.polyfitlanes) {
        // LINE_STRIP 마커
        visualization_msgs::msg::Marker line_marker;
        line_marker.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
        line_marker.header.stamp = current_time;
        line_marker.ns = "polylanes_" + poly_lane.id;
        line_marker.id = polyfit_marker_id++;
        line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line_marker.action = visualization_msgs::msg::Marker::ADD;
        line_marker.scale.x = 0.2;
        line_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

        auto it = lane_colors.find(poly_lane.id);
        if (it != lane_colors.end()) {
            line_marker.color.r = it->second[0];
            line_marker.color.g = it->second[1];
            line_marker.color.b = it->second[2];
        } else {
            line_marker.color.r = 1.0;
            line_marker.color.g = 1.0;
            line_marker.color.b = 1.0;
        }
        line_marker.color.a = 1.0;

        const double x_start = -30.0;
        const double x_end = 20.0;
        const double step = 0.5;

        for (double x = x_start; x <= x_end; x += step) {
            double y = poly_lane.a0 + poly_lane.a1 * x
                     + poly_lane.a2 * x * x
                     + poly_lane.a3 * x * x * x;
            geometry_msgs::msg::Point p;
            p.x = x;
            p.y = y;
            p.z = 0.05;
            line_marker.points.push_back(p);
        }

        polyfit_lanes_marker_array.markers.push_back(line_marker);
    }

    p_polyfit_lanes_marker_->publish(polyfit_lanes_marker_array);

    // ===== DrivingWay Marker 별도 publish (RViz 호환용) =====
    visualization_msgs::msg::MarkerArray driving_way_marker_array;

    // ego_center_lane만 표시
    visualization_msgs::msg::Marker driveway_line;
    driveway_line.header.frame_id = lane_points.frame_id.empty() ? "vehicle" : lane_points.frame_id;
    driveway_line.header.stamp = current_time;
    driveway_line.ns = "driveway";
    driveway_line.id = 0;
    driveway_line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    driveway_line.action = visualization_msgs::msg::Marker::ADD;
    driveway_line.scale.x = 0.3;  // 더 두껍게
    driveway_line.lifetime = rclcpp::Duration::from_seconds(0.5);
    driveway_line.color.r = 0.0;
    driveway_line.color.g = 1.0;
    driveway_line.color.b = 0.0;
    driveway_line.color.a = 1.0;

    const double x_start = -30.0;
    const double x_end = 20.0;
    const double step = 0.5;

    for (double x = x_start; x <= x_end; x += step) {
        double y = driving_way.a0 + driving_way.a1 * x
                 + driving_way.a2 * x * x
                 + driving_way.a3 * x * x * x;
        geometry_msgs::msg::Point p;
        p.x = x;
        p.y = y;
        p.z = 0.1;
        driveway_line.points.push_back(p);
    }

    driving_way_marker_array.markers.push_back(driveway_line);
    p_driving_way_marker_->publish(driving_way_marker_array);
}

int main(int argc, char **argv) {
    std::string node_name = "autonomous_driving";

    // Initialize node
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AutonomousDriving>(node_name));
    rclcpp::shutdown();
    return 0;
}
