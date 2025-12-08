#ifndef LANE_PROCESSOR_HPP
#define LANE_PROCESSOR_HPP

#include <vector>
#include <geometry_msgs/msg/point.hpp>
#include "interface_lane.hpp"
#include "interface_vehicle.hpp"

namespace lane_processor {

// 상태 머신 정의
enum class LaneTrackingState {
    INIT,       // 시뮬레이터 시작 직후 또는 HardReset 직후 (total_processed_frames_ < INITIALIZATION_FRAMES)
    NORMAL,     // 정상 동작: 기존 로직으로 lane1~4 관리
    RECOVERY    // 에러 발생 후, 메모리를 초기화하고 다시 천천히 레인을 재학습하는 상태
};

class LaneProcessor {
public:
    // 초기화 단계 프레임 수 (1~2프레임)
    static constexpr int INITIALIZATION_FRAMES = 2;

    // RECOVERY 모드에서 NORMAL로 전환하기 위한 안정 프레임 수
    static constexpr int RECOVERY_STABLE_FRAMES = 10;

    // 에러 검출용 상수: 차선 폭 범위
    static constexpr double MIN_LANE_WIDTH = 1.5;
    static constexpr double MAX_LANE_WIDTH = 6.0;

    /// HardReset 쿨다운 시간 (초). 연속 에러 시에도 이 간격보다 자주 리셋되지 않음
    static constexpr double HARD_RESET_COOLDOWN_SEC = 1.0;

    // RECOVERY 모드 polyfit 생성을 위한 최소 포인트/x범위
    static constexpr size_t RECOVERY_MIN_POINTS = 10;
    static constexpr double RECOVERY_MIN_X_RANGE = 5.0;

    // 메모리 저장 범위: x 최솟값 (차량 뒤쪽)
    static constexpr double MEMORY_X_MIN = -15.0;

    // 메모리 저장 범위: x 최댓값 (차량 앞쪽)
    static constexpr double MEMORY_X_MAX = 20.0;

    // 각 레인당 최대 포인트 개수
    static constexpr size_t MAX_POINTS_PER_LANE = 100;

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

    void FitLanePolynomials(const interface::VehicleState& vehicle_state);

    // 기존 폴리핏을 사용하지 않고, 현재 lane*_points_ 메모리만으로
    // lane1~lane4의 폴리핏을 "지금 차량 좌표계 기준"으로 한 번 갱신하는 함수
    void UpdatePolyfitsFromMemoryOnly();

    void GenerateEgoCenterLane();

    void DetermineCurrentDriveway();

    // Lane ordering 안전장치
    bool EnforceLaneOrdering();

    // 이전 polyfit 저장 (롤백용)
    void SavePreviousPolyfits();

    // Driveway 변경 시 outer lane 재구성
    void RebuildOuterLanesOnDrivewayChange();

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

    // 에러 감지 함수: 차선 순서/폭/NaN 체크
    bool HasLaneDetectionError();

    // 전체 메모리 초기화 및 RECOVERY 모드 진입
    void HardReset();

    // NORMAL 모드 처리 (기존 로직)
    interface::PolyfitLanes ProcessNormalMode(
        const interface::Lane& input_lane_data,
        const interface::VehicleState& vehicle_state);

    // RECOVERY 모드 처리
    interface::PolyfitLanes ProcessRecoveryMode(
        const interface::Lane& input_lane_data,
        const interface::VehicleState& vehicle_state);

    // ========== Helper 함수 (람다 대체) ==========

    // Ego-motion compensation: 포인트 배열을 이전 vehicle frame에서 현재 frame으로 변환
    void CompensateLanePoints(
        std::vector<interface::Point2D>& pts,
        const interface::VehicleState& from_state,
        const interface::VehicleState& to_state);

    // X 범위로 포인트 필터링 (MEMORY_X_MIN ~ MEMORY_X_MAX)
    void FilterPointsByXRange(std::vector<interface::Point2D>& pts);

    // X/Y 범위로 포인트 필터링 (MEMORY_X_MIN ~ MEMORY_X_MAX, -max_y ~ +max_y)
    void FilterLanePointsByXYRange(std::vector<interface::Point2D>& pts, double max_y);

    // 단일 레인 다운샘플링 (MAX_POINTS_PER_LANE 초과 시)
    void DownsampleSingleLane(std::vector<interface::Point2D>& pts);

    // 단일 레인 다운샘플링 (트리거 팩터 적용, 1.5배 초과 시만)
    void DownsampleSingleLaneWithTrigger(std::vector<interface::Point2D>& pts);

    // 포인트 배열의 x 범위 계산
    double ComputeXRange(
        const std::vector<interface::Point2D>& pts,
        double& out_x_min,
        double& out_x_max) const;

    // Polyfit 계수가 유한한지 체크
    bool CheckFinitePolyfit(const interface::PolyfitLane& poly) const;

    // 두 레인이 너무 가까운지 (겹침) 체크
    bool AreLanesTooClose(
        const interface::PolyfitLane& a,
        const interface::PolyfitLane& b) const;

    // 단일 레인 피팅 수행
    void FitSingleLane(
        std::vector<interface::Point2D>& pts,
        interface::PolyfitLane& poly,
        bool& has_real_points,
        bool relax_condition);

    // 이전 polyfit 샘플과 혼합하여 refit
    bool RefitWithSamples(
        const std::vector<interface::Point2D>& pts,
        interface::PolyfitLane& poly);

    // Driveway 후보 추가 (폭 검사 포함)
    bool TryAddDrivewayCandidate(
        int id,
        const interface::PolyfitLane& left,
        const interface::PolyfitLane& right,
        double ref_x,
        int& out_best_id,
        double& out_best_abs_center) const;

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

    // ========== 상태 머신 관련 멤버 ==========
    LaneTrackingState tracking_state_;

    // RECOVERY 모드에서 사용할 가상 레인
    interface::PolyfitLane virtual_left_lane_;
    interface::PolyfitLane virtual_right_lane_;
    bool virtual_left_valid_;
    bool virtual_right_valid_;
    int recovery_frames_;
    int recovery_stable_count_;  // 연속으로 안정된 프레임 수

    // RECOVERY 모드에서 사용할 포인트 버퍼
    std::vector<interface::Point2D> recovery_left_points_;
    std::vector<interface::Point2D> recovery_right_points_;

    // 에러 카운터 (히스테리시스 처리용)
    int consecutive_soft_error_frames_;
    int consecutive_fatal_error_frames_;

    /// 마지막 HardReset이 실행된 시각 (초, steady_clock 기준)
    double last_hard_reset_time_;
};

}  // namespace lane_processor

#endif  // LANE_PROCESSOR_HPP
