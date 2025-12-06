#ifndef LANE_PROCESSOR_HPP
#define LANE_PROCESSOR_HPP

#include <vector>
#include <geometry_msgs/msg/point.hpp>
#include "interface_lane.hpp"
#include "interface_vehicle.hpp"

namespace lane_processor {

class LaneProcessor {
public:
    // 초기화 단계 프레임 수 (1~2프레임)
    static constexpr int INITIALIZATION_FRAMES = 2;

    // 메모리 저장 범위: x 최솟값 (차량 뒤쪽)
    static constexpr double MEMORY_X_MIN = -15.0;

    // 메모리 저장 범위: x 최댓값 (차량 앞쪽)
    static constexpr double MEMORY_X_MAX = 20.0;

    // 각 레인당 최대 포인트 개수
    static constexpr size_t MAX_POINTS_PER_LANE = 50;

    LaneProcessor();

    interface::PolyfitLanes Process(const interface::Lane& input_lane_data,
                                     const interface::VehicleState& vehicle_state);

    const std::vector<interface::Point2D>& GetLane1Points() const;
    const std::vector<interface::Point2D>& GetLane2Points() const;
    const std::vector<interface::Point2D>& GetLane3Points() const;
    const std::vector<interface::Point2D>& GetLane4Points() const;

    const interface::PolyfitLane& GetLane1Polyfit() const;
    const interface::PolyfitLane& GetLane2Polyfit() const;
    const interface::PolyfitLane& GetLane3Polyfit() const;
    const interface::PolyfitLane& GetLane4Polyfit() const;

    const interface::PolyfitLane& GetEgoLane() const;

    int GetCurrentDriveway() const;

    std::vector<geometry_msgs::msg::Point> getLane1PointsMemoryVehicleFrame(
        const interface::VehicleState& vehicle_state) const;
    std::vector<geometry_msgs::msg::Point> getLane2PointsMemoryVehicleFrame(
        const interface::VehicleState& vehicle_state) const;
    std::vector<geometry_msgs::msg::Point> getLane3PointsMemoryVehicleFrame(
        const interface::VehicleState& vehicle_state) const;
    std::vector<geometry_msgs::msg::Point> getLane4PointsMemoryVehicleFrame(
        const interface::VehicleState& vehicle_state) const;

private:
    void ClassifyAndStoreLanesByY(
        const std::vector<interface::Point2D>& input_points,
        const interface::VehicleState& vehicle_state);

    void ClassifyAndStoreLanesByPolyfit(
        const std::vector<interface::Point2D>& input_points,
        const interface::VehicleState& vehicle_state);

    void FilterStoredPointsByRange(const interface::VehicleState& vehicle_state);

    void DownsampleLanePoints();

    void FitLanePolynomials(const interface::VehicleState& vehicle_state, bool should_log);

    void GenerateEgoCenterLane(bool should_log);

    void GenerateLane1And4ByExtrapolation();

    void DetermineCurrentDriveway();

    // Lane ordering 안전장치: y1 > y2 > y3 > y4 순서가 깨지면 이전 polyfit으로 롤백
    bool EnforceLaneOrdering(bool should_log);

    // 이전 polyfit 저장 (롤백용)
    void SavePreviousPolyfits();

    // 이전 lane 포인트 메모리 저장 (롤백용)
    void SavePreviousLanePoints();

    // Driveway 변경 시 outer lane 재구성
    void RebuildOuterLanesOnDrivewayChange(bool should_log);

    bool FitCubicPolynomial(
        const std::vector<interface::Point2D>& points,
        interface::PolyfitLane& out_polyfit) const;

    interface::Point2D VehicleToGlobalFrame(
        const interface::Point2D& vehicle_point,
        const interface::VehicleState& vehicle_state) const;

    interface::Point2D GlobalToVehicleFrame(
        const interface::Point2D& global_point,
        const interface::VehicleState& vehicle_state) const;

    double EvaluatePolynomial(const interface::PolyfitLane& polyfit, double x) const;

    // Ego-motion compensation: 이전 vehicle frame memory를 현재 vehicle frame으로 변환
    void ApplyEgoMotionCompensation(const interface::VehicleState& current_state);

    // 외분 유틸리티 (1:1 외분)
    void ExtrapolateLane(const interface::PolyfitLane& base,
                         const interface::PolyfitLane& ref,
                         interface::PolyfitLane& target);

    // 외분 유틸리티 (2:1 외분)
    void ExtrapolateLane2(const interface::PolyfitLane& base,
                          const interface::PolyfitLane& ref,
                          interface::PolyfitLane& target);

    // 멤버 변수
    int current_driveway_;
    int prev_driveway_;  // 이전 프레임의 driveway (레인 체인지 감지용)
    int total_processed_frames_;

    interface::VehicleState prev_vehicle_state_;  // 이전 프레임의 vehicle state (ego-motion 계산용)

    std::vector<interface::Point2D> lane1_points_;  // lane1 포인트들 - 차량 좌표계 (ego-motion compensated)
    std::vector<interface::Point2D> lane2_points_;  // lane2 포인트들 - 차량 좌표계 (ego-motion compensated)
    std::vector<interface::Point2D> lane3_points_;  // lane3 포인트들 - 차량 좌표계 (ego-motion compensated)
    std::vector<interface::Point2D> lane4_points_;  // lane4 포인트들 - 차량 좌표계 (ego-motion compensated)

    interface::PolyfitLane lane1_polyfit_;  // lane1 다항식
    interface::PolyfitLane lane2_polyfit_;  // lane2 다항식
    interface::PolyfitLane lane3_polyfit_;  // lane3 다항식
    interface::PolyfitLane lane4_polyfit_;  // lane4 다항식

    interface::PolyfitLane ego_center_lane_;  // 현재 driveway 중심선

    double last_log_time_;  // 마지막 로그 출력 시간

    // 실측 기반 여부 추적 (외분/내분으로만 만들어졌는지)
    bool lane1_is_generated_;
    bool lane2_is_generated_;
    bool lane3_is_generated_;
    bool lane4_is_generated_;

    // 이전 polyfit 저장 (롤백용)
    interface::PolyfitLane prev_lane1_polyfit_;
    interface::PolyfitLane prev_lane2_polyfit_;
    interface::PolyfitLane prev_lane3_polyfit_;
    interface::PolyfitLane prev_lane4_polyfit_;

    // 이전 프레임의 lane 포인트 메모리 (롤백용)
    std::vector<interface::Point2D> prev_lane1_points_;
    std::vector<interface::Point2D> prev_lane2_points_;
    std::vector<interface::Point2D> prev_lane3_points_;
    std::vector<interface::Point2D> prev_lane4_points_;
};

}  // namespace lane_processor

#endif  // LANE_PROCESSOR_HPP
