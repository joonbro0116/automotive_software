/**
 * @file lane_processor.cpp
 * @brief Lane boundary processing module for autonomous driving perception
 *
 * @date 2024
 * @author Autonomous Driving Team
 *
 * @details
 * This module is responsible for:
 *   - Lane boundary point classification and polynomial fitting
 *   - Ego driveway estimation (which lane the ego vehicle is currently in)
 *   - Ego-motion compensation (transforming stored lane points to current vehicle frame)
 *   - Hard reset and recovery logic for handling lane detection errors
 *
 * The module maintains a state machine with three states:
 *   - INIT: Initial frames where lane memory is being populated
 *   - NORMAL: Normal operation with full lane tracking
 *   - RECOVERY: Error recovery mode with simplified lane estimation
 *
 * Coordinate System:
 *   - Vehicle Frame: X-forward, Y-left, origin at vehicle center
 *   - Global Frame: World coordinates used internally for ego-motion compensation
 *
 * Lane Numbering Convention:
 *   - lane1: Leftmost lane boundary (highest Y value)
 *   - lane2: Second lane boundary from left
 *   - lane3: Third lane boundary from left
 *   - lane4: Rightmost lane boundary (lowest Y value)
 *   - Ordering invariant: y1 > y2 > y3 > y4 at any given x
 *
 * Driveway Numbering:
 *   - Driveway 1: Between lane1 and lane2
 *   - Driveway 2: Between lane2 and lane3 (default ego driveway)
 *   - Driveway 3: Between lane3 and lane4
 *
 * @note A-SPICE SWE.3 compliant module structure
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>

#include "lane_processor.hpp"

namespace lane_processor {

/**
 * @brief Comparison function for sorting points by X coordinate
 * @param point_a First point to compare
 * @param point_b Second point to compare
 * @return true if point_a.x < point_b.x
 */
static bool ComparePointsByX(const interface::Point2D& point_a,
                              const interface::Point2D& point_b) {
    return point_a.x < point_b.x;
}

// ============================================================================
// Constructor
// ============================================================================

/**
 * @brief Constructs a LaneProcessor with default initialization
 *
 * Initializes all member variables to default values:
 *   - Driveway set to 2 (center driveway)
 *   - Tracking state set to INIT
 *   - All point buffers pre-allocated for performance
 *   - Previous vehicle state initialized to origin
 */
LaneProcessor::LaneProcessor()
    : current_driveway_(2),       // 기본 driveway: 중앙 차선 (lane2-lane3 사이)
      prev_driveway_(2),
      total_processed_frames_(0),
      lane1_is_generated_(false), // 실측 기반 여부 (false = 실측, true = 외분으로 생성)
      lane2_is_generated_(false),
      lane3_is_generated_(false),
      lane4_is_generated_(false),
      lane1_is_primary_(false),   // Primary lane 여부 (매 프레임 갱신)
      lane2_is_primary_(false),
      lane3_is_primary_(false),
      lane4_is_primary_(false),
      tracking_state_(LaneTrackingState::INIT),
      virtual_left_valid_(false),
      virtual_right_valid_(false),
      recovery_frames_(0),
      recovery_stable_count_(0),
      consecutive_soft_error_frames_(0),
      consecutive_fatal_error_frames_(0),
      last_hard_reset_time_(-1e9),  // 초기값: 매우 과거 시점 (쿨다운 무시)
      hard_reset_active_(false),
      driveway_change_candidate_id_(2),
      driveway_change_candidate_count_(0)
{
    // Lane ID 설정
    lane1_polyfit_.id = "lane1";
    lane2_polyfit_.id = "lane2";
    lane3_polyfit_.id = "lane3";
    lane4_polyfit_.id = "lane4";

    // Lane point 메모리 벡터 미리 할당 (성능 최적화)
    // MAX_POINTS_PER_LANE=100이지만, 다운샘플 전 임시로 더 많이 저장될 수 있음
    static constexpr size_t INITIAL_POINT_BUFFER_CAPACITY = 1000;
    lane1_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane2_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane3_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane4_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);

    // Initialize prev_vehicle_state to origin (차량 좌표계 원점)
    prev_vehicle_state_.x = 0.0;
    prev_vehicle_state_.y = 0.0;
    prev_vehicle_state_.yaw = 0.0;
}

// ============================================================================
// Main Processing Entry Point
// ============================================================================

/**
 * @brief Main processing function called every frame
 *
 * @param input_lane_data Raw lane detection points from perception
 * @param vehicle_state Current vehicle pose (x, y, yaw in global frame)
 * @return PolyfitLanes containing fitted polynomial coefficients for all lanes
 *
 * @details Processing flow:
 *   Step 1: Apply ego-motion compensation to stored points (frame 2+)
 *   Step 2: Filter out-of-range points from memory
 *   Step 3: Dispatch to appropriate mode handler (NORMAL or RECOVERY)
 */
interface::PolyfitLanes LaneProcessor::Process(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state) {

    // -------------------------------------------------------------------------
    // Step 1: Ego-motion compensation (2프레임 이후부터 수행)
    //         이전 vehicle frame에 저장된 포인트들을 현재 vehicle frame으로 변환
    // -------------------------------------------------------------------------
    const bool is_second_frame_or_later = (total_processed_frames_ >= 1);

    if (is_second_frame_or_later) {
        // Step 1-1: 기존 포인트를 현재 차량 좌표계로 보정
        ApplyEgoMotionCompensation(vehicle_state);

        // Step 1-2: 범위 밖으로 나간 포인트만 제거 (메모리 전체 재분류 X)
        FilterStoredPointsByRange(vehicle_state);

        // Step 1-3: NORMAL/INIT 모드일 때는, 보정된 메모리를 기반으로 먼저 폴리핏을 갱신
        //           이렇게 해야 ClassifyAndStoreLanesByPolyfit()에서 사용하는 폴리핏이
        //           "현재 차량 좌표계 기준"으로 업데이트된 최신 폴리핏이 됨
        if (tracking_state_ != LaneTrackingState::RECOVERY) {
            UpdatePolyfitsFromMemoryOnly();
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: 상태에 따라 분기 처리
    //         RECOVERY 모드: 단순화된 left/right 분류로 레인 재구성
    //         INIT/NORMAL 모드: 기존 polyfit 기반 분류
    // -------------------------------------------------------------------------
    const bool is_recovery_mode = (tracking_state_ == LaneTrackingState::RECOVERY);

    if (is_recovery_mode) {
        return ProcessRecoveryMode(input_lane_data, vehicle_state);
    } else {
        // INIT 또는 NORMAL은 동일한 로직 사용
        return ProcessNormalMode(input_lane_data, vehicle_state);
    }
}

// ============================================================================
// NORMAL Mode Processing (INIT 및 정상 동작 처리)
// ============================================================================

/**
 * @brief Processes lane data in NORMAL or INIT mode
 *
 * @param input_lane_data Raw lane detection points
 * @param vehicle_state Current vehicle pose
 * @return PolyfitLanes with fitted lane polynomials
 *
 * @details Processing steps:
 *   Step 1: Classify and store new input points
 *   Step 2: Downsample lane points (limit memory usage)
 *   Step 3: Fit polynomials and determine driveway
 *   Step 4: Check for errors (NORMAL mode only)
 *   Step 5: Update state and frame counter
 */
interface::PolyfitLanes LaneProcessor::ProcessNormalMode(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state) {

    std::vector<interface::Point2D> input_points = input_lane_data.point;

    // -------------------------------------------------------------------------
    // Step 1: 신규 입력 포인트 분류하여 기존 메모리에 추가
    //         - 초기화 프레임(0~1): Y좌표 기반 단순 분류
    //         - 그 이후: 기존 polyfit 기반 분류 (driveway 중심선 기준)
    // -------------------------------------------------------------------------
    const bool is_initialization_phase =
        (total_processed_frames_ < INITIALIZATION_FRAMES);

    if (is_initialization_phase) {
        ClassifyAndStoreLanesByY(input_points, vehicle_state);
    } else {
        ClassifyAndStoreLanesByPolyfit(input_points, vehicle_state);
    }

    // -------------------------------------------------------------------------
    // Step 2: 다운샘플링 (포인트 수 제한)
    //         각 레인당 MAX_POINTS_PER_LANE(=100)개 이하로 유지
    // -------------------------------------------------------------------------
    DownsampleLanePoints();

    // -------------------------------------------------------------------------
    // Step 3: Polyfit 생성 및 driveway/center lane 갱신
    // -------------------------------------------------------------------------
    FitLanePolynomials(vehicle_state);

    // -------------------------------------------------------------------------
    // Step 4: 에러 체크 (NORMAL 모드에서만 수행)
    //         INIT 상태에서는 아직 polyfit이 불안정하므로 검사 생략
    //
    // 상태 전이 조건:
    //   NORMAL → RECOVERY: 연속 soft/fatal error 발생 시 HardReset 후 진입
    // -------------------------------------------------------------------------
    const bool is_normal_mode = (tracking_state_ == LaneTrackingState::NORMAL);

    if (is_normal_mode) {
        const bool has_error = HasLaneDetectionError();

        if (has_error) {
            HardReset();

            // HardReset 후 빈 결과 반환 (다음 프레임부터 RECOVERY 모드)
            interface::PolyfitLanes empty_result;
            empty_result.frame_id = input_lane_data.frame_id;
            empty_result.polyfitlanes.push_back(lane1_polyfit_);
            empty_result.polyfitlanes.push_back(lane2_polyfit_);
            empty_result.polyfitlanes.push_back(lane3_polyfit_);
            empty_result.polyfitlanes.push_back(lane4_polyfit_);
            empty_result.polyfitlanes.push_back(ego_center_lane_);
            return empty_result;
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: 프레임 카운터 증가 및 상태 전이
    //
    // 상태 전이 조건:
    //   INIT → NORMAL: 초기 프레임(INITIALIZATION_FRAMES=2) 동안
    //                  메모리를 채운 뒤 정상 추적 시작
    // -------------------------------------------------------------------------
    total_processed_frames_++;

    const bool should_transition_to_normal =
        (tracking_state_ == LaneTrackingState::INIT) &&
        (total_processed_frames_ >= INITIALIZATION_FRAMES);

    if (should_transition_to_normal) {
        tracking_state_ = LaneTrackingState::NORMAL;
    }

    // 다음 프레임을 위해 현재 vehicle state 저장
    prev_vehicle_state_ = vehicle_state;

    // -------------------------------------------------------------------------
    // Step 6: 결과 반환
    // -------------------------------------------------------------------------
    interface::PolyfitLanes result;
    result.frame_id = input_lane_data.frame_id;
    result.polyfitlanes.push_back(lane1_polyfit_);
    result.polyfitlanes.push_back(lane2_polyfit_);
    result.polyfitlanes.push_back(lane3_polyfit_);
    result.polyfitlanes.push_back(lane4_polyfit_);
    result.polyfitlanes.push_back(ego_center_lane_);

    return result;
}

// ============================================================================
// RECOVERY Mode Processing (에러 복구 상태 처리)
// ============================================================================

/**
 * @brief Processes lane data in RECOVERY mode
 *
 * @param input_lane_data Raw lane detection points
 * @param vehicle_state Current vehicle pose
 * @return PolyfitLanes with recovered lane polynomials
 *
 * @details
 * RECOVERY 모드에서는 단순화된 방식으로 레인을 추정:
 *   - 입력 포인트를 Y부호로만 left(Y>=0) / right(Y<0) 분류
 *   - 각 쪽의 포인트가 충분하면 virtual lane 생성
 *   - 양쪽 모두 안정화되면 NORMAL 모드로 복귀
 *
 * 상태 전이 조건:
 *   RECOVERY -> NORMAL: 가상 좌/우 레인이 RECOVERY_STABLE_FRAMES(=10)
 *                      프레임 동안 안정적으로 추정되면 복귀
 */
interface::PolyfitLanes LaneProcessor::ProcessRecoveryMode(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state) {

    std::vector<interface::Point2D> input_points = input_lane_data.point;

    // -------------------------------------------------------------------------
    // Step 1: RECOVERY 버퍼에 대한 ego-motion compensation
    //         (recovery 모드 2프레임 이후부터)
    // -------------------------------------------------------------------------
    const bool is_recovery_second_frame_or_later = (recovery_frames_ >= 1);

    if (is_recovery_second_frame_or_later) {
        // 이전 vehicle frame -> 현재 vehicle frame 변환
        CompensateLanePoints(recovery_left_points_, prev_vehicle_state_, vehicle_state);
        CompensateLanePoints(recovery_right_points_, prev_vehicle_state_, vehicle_state);

        // 범위 필터링 (MEMORY_X_MIN ~ MEMORY_X_MAX)
        FilterPointsByXRange(recovery_left_points_);
        FilterPointsByXRange(recovery_right_points_);
    }

    // -------------------------------------------------------------------------
    // Step 2: 입력 포인트를 left/right로 분류 (Y 부호 기준)
    //         Y >= 0 -> left (vehicle 기준 왼쪽)
    //         Y <  0 -> right (vehicle 기준 오른쪽)
    // -------------------------------------------------------------------------
    for (const auto& input_pt : input_points) {
        // X 범위 필터링
        const bool is_x_in_memory_range =
            (input_pt.x >= MEMORY_X_MIN) && (input_pt.x <= MEMORY_X_MAX);

        if (!is_x_in_memory_range) {
            continue;
        }

        const bool is_left_side = (input_pt.y >= 0.0);

        if (is_left_side) {
            recovery_left_points_.push_back(input_pt);
        } else {
            recovery_right_points_.push_back(input_pt);
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: 다운샘플링 (각 버퍼별로)
    // -------------------------------------------------------------------------
    DownsampleSingleLane(recovery_left_points_);
    DownsampleSingleLane(recovery_right_points_);

    // -------------------------------------------------------------------------
    // Step 4: 각 쪽의 포인트가 충분하면 virtual lane 생성
    // -------------------------------------------------------------------------
    double left_x_min = 0.0;
    double left_x_max = 0.0;
    double right_x_min = 0.0;
    double right_x_max = 0.0;

    const double left_x_range =
        ComputeXRange(recovery_left_points_, left_x_min, left_x_max);
    const double right_x_range =
        ComputeXRange(recovery_right_points_, right_x_min, right_x_max);

    // Left side 유효성 체크
    const bool left_has_enough_points =
        (recovery_left_points_.size() >= RECOVERY_MIN_POINTS);
    const bool left_has_enough_range =
        (left_x_range >= RECOVERY_MIN_X_RANGE);
    const bool left_is_valid = left_has_enough_points && left_has_enough_range;

    // Right side 유효성 체크
    const bool right_has_enough_points =
        (recovery_right_points_.size() >= RECOVERY_MIN_POINTS);
    const bool right_has_enough_range =
        (right_x_range >= RECOVERY_MIN_X_RANGE);
    const bool right_is_valid = right_has_enough_points && right_has_enough_range;

    // Virtual lane 피팅
    if (left_is_valid) {
        const bool fit_success =
            FitCubicPolynomial(recovery_left_points_, virtual_left_lane_);
        if (fit_success) {
            virtual_left_valid_ = true;
        }
    }

    if (right_is_valid) {
        const bool fit_success =
            FitCubicPolynomial(recovery_right_points_, virtual_right_lane_);
        if (fit_success) {
            virtual_right_valid_ = true;
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: 현재 프레임의 출력 구성
    //         - virtual left -> lane2 (ego 왼쪽 경계)
    //         - virtual right -> lane3 (ego 오른쪽 경계)
    //         - lane1, lane4는 외분으로 생성
    // -------------------------------------------------------------------------
    if (virtual_left_valid_) {
        lane2_polyfit_ = virtual_left_lane_;
        lane2_polyfit_.id = "lane2";
    }

    if (virtual_right_valid_) {
        lane3_polyfit_ = virtual_right_lane_;
        lane3_polyfit_.id = "lane3";
    }

    // 양쪽 모두 유효한 경우에만 outer lane과 center lane 생성
    const bool both_sides_valid = virtual_left_valid_ && virtual_right_valid_;

    if (both_sides_valid) {
        // lane1 = lane2 기준 왼쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_polyfit_.id = "lane1";
        lane1_is_generated_ = true;

        // lane4 = lane3 기준 오른쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_polyfit_.id = "lane4";
        lane4_is_generated_ = true;

        // Ego center lane 생성 (lane2와 lane3의 평균)
        ego_center_lane_.id = "ego_center";
        ego_center_lane_.frame_id = "vehicle";
        ego_center_lane_.a0 = (lane2_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane2_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane2_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane2_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);
    }

    // -------------------------------------------------------------------------
    // Step 6: Recovery 프레임 카운터 증가
    // -------------------------------------------------------------------------
    recovery_frames_++;

    // -------------------------------------------------------------------------
    // Step 7: 안정성 체크 및 NORMAL 모드 전환 결정
    //         양쪽 레인이 유효하면 stable count 증가, 아니면 리셋
    //
    // 상태 전이 조건:
    //   RECOVERY -> NORMAL: recovery_stable_count_ >= RECOVERY_STABLE_FRAMES
    // -------------------------------------------------------------------------
    if (both_sides_valid) {
        recovery_stable_count_++;
    } else {
        recovery_stable_count_ = 0;
    }

    const bool recovery_is_stable =
        (recovery_stable_count_ >= RECOVERY_STABLE_FRAMES);

    if (recovery_is_stable) {
        // Recovery 포인트를 lane2, lane3 버퍼로 이전
        lane2_points_ = recovery_left_points_;
        lane3_points_ = recovery_right_points_;

        lane2_is_generated_ = false;
        lane3_is_generated_ = false;

        // Driveway를 기본값(2)으로 설정
        current_driveway_ = 2;
        prev_driveway_ = 2;

        // NORMAL 모드로 전환
        tracking_state_ = LaneTrackingState::NORMAL;

        std::cout << "[LaneProcessor] State transition: RECOVERY -> NORMAL"
                  << std::endl;

        // HardReset 락 해제 및 에러 카운터 리셋
        hard_reset_active_ = false;
        consecutive_soft_error_frames_ = 0;
        consecutive_fatal_error_frames_ = 0;

        // RECOVERY 버퍼 초기화
        recovery_left_points_.clear();
        recovery_right_points_.clear();
        virtual_left_valid_ = false;
        virtual_right_valid_ = false;
        recovery_frames_ = 0;
        recovery_stable_count_ = 0;
    }

    // -------------------------------------------------------------------------
    // Step 8: 프레임 카운터 증가 및 vehicle state 저장
    // -------------------------------------------------------------------------
    total_processed_frames_++;
    prev_vehicle_state_ = vehicle_state;

    // -------------------------------------------------------------------------
    // Step 9: 결과 반환
    // -------------------------------------------------------------------------
    interface::PolyfitLanes result;
    result.frame_id = input_lane_data.frame_id;
    result.polyfitlanes.push_back(lane1_polyfit_);
    result.polyfitlanes.push_back(lane2_polyfit_);
    result.polyfitlanes.push_back(lane3_polyfit_);
    result.polyfitlanes.push_back(lane4_polyfit_);
    result.polyfitlanes.push_back(ego_center_lane_);

    return result;
}

// ============================================================================
// Ego-motion Compensation (세계좌표로 보관된 차선 포인트를 현재 vehicle frame으로 변환)
// ============================================================================

/**
 * @brief Applies ego-motion compensation to all lane point buffers
 *
 * @param current_state Current vehicle pose in global frame
 *
 * @details
 * 좌표 변환 과정:
 *   1. 이전 vehicle frame -> global frame (prev_vehicle_state_ 기준)
 *   2. global frame -> 현재 vehicle frame (current_state 기준)
 *
 * 이를 통해 차량이 이동해도 메모리에 저장된 포인트들이
 * 현재 차량 좌표계에서 올바른 위치를 유지함
 */
void LaneProcessor::ApplyEgoMotionCompensation(
    const interface::VehicleState& current_state) {

    // 실측 포인트 compensation
    CompensateLanePoints(lane1_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane2_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane3_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane4_points_, prev_vehicle_state_, current_state);

    // 샘플 포인트 compensation
    CompensateLanePoints(lane1_sample_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane2_sample_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane3_sample_points_, prev_vehicle_state_, current_state);
    CompensateLanePoints(lane4_sample_points_, prev_vehicle_state_, current_state);
}

/**
 * @brief Transforms lane points from one vehicle frame to another
 *
 * @param[in,out] lane_points Points to transform (modified in place)
 * @param[in] from_state Previous vehicle pose (source frame)
 * @param[in] to_state Current vehicle pose (target frame)
 *
 * @details
 * 변환 순서:
 *   1. 이전 vehicle frame -> global frame
 *      global_x = from_state.x + cos(from_yaw)*veh_x - sin(from_yaw)*veh_y
 *      global_y = from_state.y + sin(from_yaw)*veh_x + cos(from_yaw)*veh_y
 *
 *   2. global frame -> 현재 vehicle frame
 *      dx = global_x - to_state.x
 *      dy = global_y - to_state.y
 *      new_veh_x = cos(-to_yaw)*dx - sin(-to_yaw)*dy
 *      new_veh_y = sin(-to_yaw)*dx + cos(-to_yaw)*dy
 */
void LaneProcessor::CompensateLanePoints(
    std::vector<interface::Point2D>& lane_points,
    const interface::VehicleState& from_state,
    const interface::VehicleState& to_state) {

    // 삼각함수 값 미리 계산 (성능 최적화)
    const double cos_from_yaw = std::cos(from_state.yaw);
    const double sin_from_yaw = std::sin(from_state.yaw);
    const double cos_neg_to_yaw = std::cos(-to_state.yaw);
    const double sin_neg_to_yaw = std::sin(-to_state.yaw);

    for (auto& point : lane_points) {
        // Step 1: 이전 vehicle frame -> global frame
        const double global_x =
            from_state.x + cos_from_yaw * point.x - sin_from_yaw * point.y;
        const double global_y =
            from_state.y + sin_from_yaw * point.x + cos_from_yaw * point.y;

        // Step 2: global frame -> 현재 vehicle frame
        const double delta_x = global_x - to_state.x;
        const double delta_y = global_y - to_state.y;

        point.x = cos_neg_to_yaw * delta_x - sin_neg_to_yaw * delta_y;
        point.y = sin_neg_to_yaw * delta_x + cos_neg_to_yaw * delta_y;
    }
}

// ============================================================================
// Lane Point Classification - Y-based (초기 프레임용 단순 분류)
// ============================================================================

/**
 * @brief Classifies input points by Y coordinate sign (초기 2프레임용)
 *
 * @param input_points Raw lane detection points in vehicle frame
 * @param vehicle_state Current vehicle state (unused in this simple classifier)
 *
 * @details
 * 초기화 단계(INITIALIZATION_FRAMES=2 프레임)에서는 아직 polyfit이 없으므로
 * 단순히 Y 좌표 부호로 분류:
 *   - Y >= 0 -> lane2 (ego 왼쪽 경계)
 *   - Y <  0 -> lane3 (ego 오른쪽 경계)
 *
 * 이 단계에서는 lane1, lane4는 채우지 않음 (나중에 외분으로 생성)
 */
void LaneProcessor::ClassifyAndStoreLanesByY(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // Unused parameter

    for (const auto& vehicle_point : input_points) {
        // X 범위 필터링 (전방 -15m ~ +20m)
        const bool is_x_in_memory_range =
            (vehicle_point.x >= MEMORY_X_MIN) && (vehicle_point.x <= MEMORY_X_MAX);

        if (!is_x_in_memory_range) {
            continue;
        }

        // Y 부호에 따라 분류
        const bool is_left_side = (vehicle_point.y >= 0.0);

        if (is_left_side) {
            lane2_points_.push_back(vehicle_point);
        } else {
            lane3_points_.push_back(vehicle_point);
        }
    }
}

// ============================================================================
// Lane Point Classification - Polyfit-based (3프레임 이후 사용)
// ============================================================================

/**
 * @brief Classifies input points based on existing polyfit lane centers
 *
 * @param input_points Raw lane detection points in vehicle frame
 * @param vehicle_state Current vehicle state (unused)
 *
 * @details
 * 3프레임 이후에는 기존 polyfit을 기준으로 분류:
 *   1. 각 driveway(1~3)의 중심선 Y값 계산
 *   2. 입력 포인트와 각 중심선 거리 비교
 *   3. 가장 가까운 driveway의 중심선 기준으로 좌/우 결정
 *   4. 해당 레인 버퍼에 포인트 추가
 *
 * 이 방식은 차선 변경 시에도 정확한 분류를 가능하게 함
 */
void LaneProcessor::ClassifyAndStoreLanesByPolyfit(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // Unused parameter

    for (const auto& vehicle_point : input_points) {
        const double point_x = vehicle_point.x;
        const double point_y = vehicle_point.y;

        // X 범위 필터링 (전방 -15m ~ +20m)
        const bool is_x_in_memory_range =
            (point_x >= MEMORY_X_MIN) && (point_x <= MEMORY_X_MAX);

        if (!is_x_in_memory_range) {
            continue;
        }

        // ---------------------------------------------------------------------
        // Step 1: 현재 x 위치에서 각 lane의 Y값 계산
        // ---------------------------------------------------------------------
        const double lane1_y_at_x = EvaluatePolynomial(lane1_polyfit_, point_x);
        const double lane2_y_at_x = EvaluatePolynomial(lane2_polyfit_, point_x);
        const double lane3_y_at_x = EvaluatePolynomial(lane3_polyfit_, point_x);
        const double lane4_y_at_x = EvaluatePolynomial(lane4_polyfit_, point_x);

        // ---------------------------------------------------------------------
        // Step 2: 각 driveway 후보 정의
        //         Driveway N의 중심선 = (lane_N + lane_{N+1}) / 2
        // ---------------------------------------------------------------------
        struct DrivewayCandidate {
            int left_lane_index;
            int right_lane_index;
            double center_y;
            double distance_to_point;
        };

        DrivewayCandidate candidates[3];

        // Driveway 1: lane1 (left boundary) ~ lane2 (right boundary)
        candidates[0].left_lane_index = 1;
        candidates[0].right_lane_index = 2;
        candidates[0].center_y = 0.5 * (lane1_y_at_x + lane2_y_at_x);
        candidates[0].distance_to_point = std::fabs(point_y - candidates[0].center_y);

        // Driveway 2: lane2 (left boundary) ~ lane3 (right boundary)
        candidates[1].left_lane_index = 2;
        candidates[1].right_lane_index = 3;
        candidates[1].center_y = 0.5 * (lane2_y_at_x + lane3_y_at_x);
        candidates[1].distance_to_point = std::fabs(point_y - candidates[1].center_y);

        // Driveway 3: lane3 (left boundary) ~ lane4 (right boundary)
        candidates[2].left_lane_index = 3;
        candidates[2].right_lane_index = 4;
        candidates[2].center_y = 0.5 * (lane3_y_at_x + lane4_y_at_x);
        candidates[2].distance_to_point = std::fabs(point_y - candidates[2].center_y);

        // ---------------------------------------------------------------------
        // Step 3: 가장 가까운 driveway 선택 (threshold 없이 순수 거리 비교)
        // ---------------------------------------------------------------------
        const DrivewayCandidate* best_candidate = &candidates[0];
        for (int candidate_idx = 1; candidate_idx < 3; ++candidate_idx) {
            if (candidates[candidate_idx].distance_to_point <
                best_candidate->distance_to_point) {
                best_candidate = &candidates[candidate_idx];
            }
        }

        // ---------------------------------------------------------------------
        // Step 4: 선택된 driveway의 중앙선 기준으로 좌/우 결정
        //         point_y > center_y -> 왼쪽 레인
        //         point_y <= center_y -> 오른쪽 레인
        // ---------------------------------------------------------------------
        const bool is_point_left_of_center = (point_y > best_candidate->center_y);
        const int assigned_lane_index = is_point_left_of_center
            ? best_candidate->left_lane_index
            : best_candidate->right_lane_index;

        // ---------------------------------------------------------------------
        // Step 5: 해당 레인 버퍼에 포인트 추가
        // ---------------------------------------------------------------------
        switch (assigned_lane_index) {
            case 1:
                lane1_points_.push_back(vehicle_point);
                break;
            case 2:
                lane2_points_.push_back(vehicle_point);
                break;
            case 3:
                lane3_points_.push_back(vehicle_point);
                break;
            case 4:
                lane4_points_.push_back(vehicle_point);
                break;
            default:
                // Invalid lane index - should not happen
                break;
        }
    }
}

// ============================================================================
// Extrapolation Utilities (외분을 통한 레인 생성)
// ============================================================================

/**
 * @brief Extrapolates a lane using 1:1 ratio (한 칸 외분)
 *
 * @param base_lane Base lane polynomial (extrapolation reference)
 * @param reference_lane Reference lane polynomial
 * @param[out] target_lane Output extrapolated lane polynomial
 *
 * @details
 * 1:1 외분 공식: target = 2 * base - reference
 *
 * 사용 예:
 *   - lane1 생성: ExtrapolateLane(lane2, lane3, lane1) -> lane1 = 2*lane2 - lane3
 *   - lane4 생성: ExtrapolateLane(lane3, lane2, lane4) -> lane4 = 2*lane3 - lane2
 *
 * X 범위는 두 입력 레인의 교집합으로 설정 (overlap이 없으면 합집합 사용)
 */
void LaneProcessor::ExtrapolateLane(
    const interface::PolyfitLane& base_lane,
    const interface::PolyfitLane& reference_lane,
    interface::PolyfitLane& target_lane) {

    // 1:1 외분: target = 2*base - ref
    target_lane.a0 = 2.0 * base_lane.a0 - reference_lane.a0;
    target_lane.a1 = 2.0 * base_lane.a1 - reference_lane.a1;
    target_lane.a2 = 2.0 * base_lane.a2 - reference_lane.a2;
    target_lane.a3 = 2.0 * base_lane.a3 - reference_lane.a3;

    // X 범위 결정: 두 입력의 교집합 시도
    double x_range_start = std::max(base_lane.x_start, reference_lane.x_start);
    double x_range_end = std::min(base_lane.x_end, reference_lane.x_end);

    // 교집합이 없으면 합집합 사용
    const bool no_overlap = (x_range_end <= x_range_start);
    if (no_overlap) {
        x_range_start = std::min(base_lane.x_start, reference_lane.x_start);
        x_range_end = std::max(base_lane.x_end, reference_lane.x_end);
    }

    target_lane.x_start = x_range_start;
    target_lane.x_end = x_range_end;
}

/**
 * @brief Extrapolates a lane using 2:1 ratio (두 칸 외분)
 *
 * @param base_lane Base lane polynomial (extrapolation reference)
 * @param reference_lane Reference lane polynomial
 * @param[out] target_lane Output extrapolated lane polynomial
 *
 * @details
 * 2:1 외분 공식: target = 3 * base - 2 * reference
 *
 * 사용 예:
 *   - Driveway 1에서 lane4 생성: lane4 = 3*lane2 - 2*lane1
 *   - Driveway 3에서 lane1 생성: lane1 = 3*lane3 - 2*lane4
 *
 * 이 공식은 base 레인에서 reference 반대 방향으로 2칸 떨어진 위치를 추정
 */
void LaneProcessor::ExtrapolateLane2(
    const interface::PolyfitLane& base_lane,
    const interface::PolyfitLane& reference_lane,
    interface::PolyfitLane& target_lane) {

    // 2:1 외분: target = 3*base - 2*ref
    target_lane.a0 = 3.0 * base_lane.a0 - 2.0 * reference_lane.a0;
    target_lane.a1 = 3.0 * base_lane.a1 - 2.0 * reference_lane.a1;
    target_lane.a2 = 3.0 * base_lane.a2 - 2.0 * reference_lane.a2;
    target_lane.a3 = 3.0 * base_lane.a3 - 2.0 * reference_lane.a3;

    // X 범위 결정: 두 입력의 교집합 시도
    double x_range_start = std::max(base_lane.x_start, reference_lane.x_start);
    double x_range_end = std::min(base_lane.x_end, reference_lane.x_end);

    // 교집합이 없으면 합집합 사용
    const bool no_overlap = (x_range_end <= x_range_start);
    if (no_overlap) {
        x_range_start = std::min(base_lane.x_start, reference_lane.x_start);
        x_range_end = std::max(base_lane.x_end, reference_lane.x_end);
    }

    target_lane.x_start = x_range_start;
    target_lane.x_end = x_range_end;
}

// ============================================================================
// Outer Lane Reconstruction on Driveway Change
// ============================================================================

/**
 * @brief Rebuilds outer lanes when ego driveway changes
 *
 * @details
 * Driveway 변경 시 outer lane(현재 driveway에서 멀리 있는 레인)의
 * 포인트 메모리를 초기화하고 외분으로 재생성
 *
 * Driveway별 inner/outer lane 구분:
 *   - Driveway 1: inner=[lane1,lane2], outer=[lane3,lane4]
 *   - Driveway 2: inner=[lane2,lane3], outer=[lane1,lane4]
 *   - Driveway 3: inner=[lane3,lane4], outer=[lane1,lane2]
 *
 * 외분 비율:
 *   - 인접 outer lane: 1:1 외분
 *   - 2칸 떨어진 outer lane: 2:1 외분
 */
void LaneProcessor::RebuildOuterLanesOnDrivewayChange() {
    // Driveway 변경이 없으면 아무것도 하지 않음
    const bool driveway_unchanged = (prev_driveway_ == current_driveway_);
    if (driveway_unchanged) {
        return;
    }

    std::cout << "[LaneProcessor] Driveway changed: " << prev_driveway_
              << " -> " << current_driveway_ << std::endl;

    if (current_driveway_ == 1) {
        // ---------------------------------------------------------------------
        // Driveway 1: ego 주변은 lane1, lane2 유지
        //             outer는 lane3, lane4 재구성
        // ---------------------------------------------------------------------
        lane3_points_.clear();
        lane4_points_.clear();

        // lane3 = lane2 기준 오른쪽 한 칸 (1:1 외분: lane3 = 2*lane2 - lane1)
        ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
        lane3_is_generated_ = true;

        // lane4 = lane2 기준 오른쪽 두 칸 (2:1 외분: lane4 = 3*lane2 - 2*lane1)
        ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

    } else if (current_driveway_ == 2) {
        // ---------------------------------------------------------------------
        // Driveway 2: ego 주변은 lane2, lane3 유지
        //             outer는 lane1, lane4 재구성
        // ---------------------------------------------------------------------
        lane1_points_.clear();
        lane4_points_.clear();

        // lane1 = lane2 기준 왼쪽 한 칸 (1:1 외분: lane1 = 2*lane2 - lane3)
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;

        // lane4 = lane3 기준 오른쪽 한 칸 (1:1 외분: lane4 = 2*lane3 - lane2)
        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

    } else if (current_driveway_ == 3) {
        // ---------------------------------------------------------------------
        // Driveway 3: ego 주변은 lane3, lane4 유지
        //             outer는 lane1, lane2 재구성
        // ---------------------------------------------------------------------
        lane1_points_.clear();
        lane2_points_.clear();

        // lane2 = lane3 기준 왼쪽 한 칸 (1:1 외분: lane2 = 2*lane3 - lane4)
        ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
        lane2_is_generated_ = true;

        // lane1 = lane3 기준 왼쪽 두 칸 (2:1 외분: lane1 = 3*lane3 - 2*lane4)
        ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;
    }
}

// ============================================================================
// Update Polyfits From Memory Only (ego-motion 보정 후 폴리핏 선갱신)
// ============================================================================

/**
 * @brief Updates lane polyfits using only current lane point buffers
 *
 * @details
 * 이 함수는 ego-motion 보정이 끝난 lane*_points_만을 사용해서
 * lane1~lane4 폴리핏을 "현재 차량 좌표계" 기준으로 한 번 갱신하는 역할을 한다.
 *
 * 이 함수에서는 다음을 수행하지 않는다:
 *   - DetermineCurrentDriveway() 호출
 *   - RebuildOuterLanesOnDrivewayChange() 호출
 *   - prev_lane*_polyfit_ 관련 저장/롤백 로직
 *   - generated 플래그 갱신
 *   - ego_center_lane_ 갱신
 *
 * 목적:
 *   ClassifyAndStoreLanesByPolyfit()에서 사용할 폴리핏이
 *   "이전 프레임 기준"이 아닌 "현재 프레임 기준"이 되도록 선갱신
 */
void LaneProcessor::UpdatePolyfitsFromMemoryOnly() {
    // 이 함수는 ego-motion 보정이 끝난 lane*_points_만을 사용해서
    // lane1~lane4 폴리핏을 "현재 차량 좌표계" 기준으로 한 번 갱신하는 역할을 한다.
    // driveway 변경/outer lane 로직은 사용하지 않는다.

    bool lane1_has_real_points = false;
    bool lane2_has_real_points = false;
    bool lane3_has_real_points = false;
    bool lane4_has_real_points = false;

    // driveway와 상관없이, 단순히 각 레인을 독립적으로 피팅한다.
    // relax_condition은 false로 설정하여 정상 기준 적용
    FitSingleLane(lane1_points_, lane1_polyfit_, lane1_has_real_points, false);
    FitSingleLane(lane2_points_, lane2_polyfit_, lane2_has_real_points, false);
    FitSingleLane(lane3_points_, lane3_polyfit_, lane3_has_real_points, false);
    FitSingleLane(lane4_points_, lane4_polyfit_, lane4_has_real_points, false);

    // 여기서는 generated 플래그나 driveway, ego_center_lane_ 등은 건드리지 않는다.
}

// ============================================================================
// Polynomial Fitting - Main Entry Point
// ============================================================================

/**
 * @brief Fits polynomials to all lane point buffers
 *
 * @param vehicle_state Current vehicle state (unused)
 *
 * @details Processing flow:
 *   Step 1: Save previous polyfits for potential rollback
 *   Step 2: Handle initialization phase (first 2 frames)
 *   Step 3: Determine current driveway
 *   Step 4: Fit inner lanes (실측 기반)
 *   Step 5: Handle outer lanes (외분 또는 실측)
 *   Step 6: Enforce lane ordering
 *   Step 7: Generate ego center lane
 */
void LaneProcessor::FitLanePolynomials(
    const interface::VehicleState& vehicle_state) {

    // -------------------------------------------------------------------------
    // Step 1: 이전 polyfit 저장 (롤백 가능성 대비)
    // -------------------------------------------------------------------------
    SavePreviousPolyfits();

    // -------------------------------------------------------------------------
    // Step 2: 초기화 프레임 처리 (0~1프레임)
    //         lane2, lane3만 피팅하고 lane1, lane4는 외분으로 생성
    // -------------------------------------------------------------------------
    const bool is_initialization_phase =
        (total_processed_frames_ < INITIALIZATION_FRAMES);

    if (is_initialization_phase) {
        bool lane2_has_points = false;
        bool lane3_has_points = false;

        // lane2, lane3 피팅 (relax_condition=false)
        FitSingleLane(lane2_points_, lane2_polyfit_, lane2_has_points, false);
        FitSingleLane(lane3_points_, lane3_polyfit_, lane3_has_points, false);

        // 양쪽 모두 포인트가 있으면 외분으로 lane1, lane4 생성
        const bool both_inner_lanes_have_points =
            lane2_has_points && lane3_has_points;

        if (both_inner_lanes_have_points) {
            // lane1 = 2*lane2 - lane3 (lane2 왼쪽)
            ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
            // lane4 = 2*lane3 - lane2 (lane3 오른쪽)
            ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        }

        // Generated 플래그 설정
        lane1_is_generated_ = true;  // 항상 외분으로 생성
        lane2_is_generated_ = !lane2_has_points;
        lane3_is_generated_ = !lane3_has_points;
        lane4_is_generated_ = true;  // 항상 외분으로 생성

        // Driveway 결정 및 저장
        DetermineCurrentDriveway();
        prev_driveway_ = current_driveway_;

        // Ego center lane 생성
        GenerateEgoCenterLane();
        return;
    }

    // -------------------------------------------------------------------------
    // Step 3: Primary lane 선정 (피팅의 핵심)
    //         신뢰 가능한 클러스터(포인트 >= RELIABLE_CLUSTER_MIN_POINTS) 중
    //         x_max가 가장 큰 두 개를 primary로 선택
    //         Primary는 Driveway와 무관하게 실측 피팅됨
    // -------------------------------------------------------------------------

    // 각 lane의 신뢰도 및 x_max 계산
    struct LaneCandidate {
        int lane_id;
        double x_max;
        bool is_reliable;
    };

    LaneCandidate candidates[4];

    // lane1
    candidates[0].lane_id = 1;
    candidates[0].is_reliable = (lane1_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS);
    candidates[0].x_max = -std::numeric_limits<double>::max();
    if (!lane1_points_.empty()) {
        for (const auto& pt : lane1_points_) {
            if (pt.x > candidates[0].x_max) {
                candidates[0].x_max = pt.x;
            }
        }
    }

    // lane2
    candidates[1].lane_id = 2;
    candidates[1].is_reliable = (lane2_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS);
    candidates[1].x_max = -std::numeric_limits<double>::max();
    if (!lane2_points_.empty()) {
        for (const auto& pt : lane2_points_) {
            if (pt.x > candidates[1].x_max) {
                candidates[1].x_max = pt.x;
            }
        }
    }

    // lane3
    candidates[2].lane_id = 3;
    candidates[2].is_reliable = (lane3_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS);
    candidates[2].x_max = -std::numeric_limits<double>::max();
    if (!lane3_points_.empty()) {
        for (const auto& pt : lane3_points_) {
            if (pt.x > candidates[2].x_max) {
                candidates[2].x_max = pt.x;
            }
        }
    }

    // lane4
    candidates[3].lane_id = 4;
    candidates[3].is_reliable = (lane4_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS);
    candidates[3].x_max = -std::numeric_limits<double>::max();
    if (!lane4_points_.empty()) {
        for (const auto& pt : lane4_points_) {
            if (pt.x > candidates[3].x_max) {
                candidates[3].x_max = pt.x;
            }
        }
    }

    // 신뢰 가능한 후보들을 x_max 기준 내림차순으로 정렬하여 상위 2개 선택
    int primary_lane_ids[2] = {-1, -1};
    int primary_count = 0;

    for (int select_idx = 0; select_idx < 2; ++select_idx) {
        int best_idx = -1;
        double best_x_max = -std::numeric_limits<double>::max();

        for (int i = 0; i < 4; ++i) {
            if (!candidates[i].is_reliable) {
                continue;
            }

            // 이미 선택된 것인지 체크
            bool already_selected = false;
            for (int j = 0; j < select_idx; ++j) {
                if (primary_lane_ids[j] == candidates[i].lane_id) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected) {
                continue;
            }

            if (candidates[i].x_max > best_x_max) {
                best_x_max = candidates[i].x_max;
                best_idx = i;
            }
        }

        if (best_idx >= 0) {
            primary_lane_ids[primary_count] = candidates[best_idx].lane_id;
            primary_count++;
        }
    }

    // Primary 플래그 설정
    lane1_is_primary_ = false;
    lane2_is_primary_ = false;
    lane3_is_primary_ = false;
    lane4_is_primary_ = false;

    for (int i = 0; i < primary_count; ++i) {
        if (primary_lane_ids[i] == 1) {
            lane1_is_primary_ = true;
        } else if (primary_lane_ids[i] == 2) {
            lane2_is_primary_ = true;
        } else if (primary_lane_ids[i] == 3) {
            lane3_is_primary_ = true;
        } else if (primary_lane_ids[i] == 4) {
            lane4_is_primary_ = true;
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: Primary lane 피팅
    //         Primary로 선정된 2개 레인은 무조건 실측 피팅
    //         FitPrimaryLaneWithMemoryAndSamples 사용
    // -------------------------------------------------------------------------
    if (lane1_is_primary_) {
        FitPrimaryLaneWithMemoryAndSamples(lane1_points_, lane1_sample_points_,
                                           lane1_polyfit_, vehicle_state);
        lane1_is_generated_ = false;
    }

    if (lane2_is_primary_) {
        FitPrimaryLaneWithMemoryAndSamples(lane2_points_, lane2_sample_points_,
                                           lane2_polyfit_, vehicle_state);
        lane2_is_generated_ = false;
    }

    if (lane3_is_primary_) {
        FitPrimaryLaneWithMemoryAndSamples(lane3_points_, lane3_sample_points_,
                                           lane3_polyfit_, vehicle_state);
        lane3_is_generated_ = false;
    }

    if (lane4_is_primary_) {
        FitPrimaryLaneWithMemoryAndSamples(lane4_points_, lane4_sample_points_,
                                           lane4_polyfit_, vehicle_state);
        lane4_is_generated_ = false;
    }

    // -------------------------------------------------------------------------
    // Step 5: Non-Primary lane 외분 생성
    //         Primary 2개의 조합에 따라 외분 방식 결정
    //         (Primary가 2개 미만인 경우는 fallback 처리)
    // -------------------------------------------------------------------------

    // Primary lane ID를 정렬하여 조합 식별 (작은 ID가 앞에)
    int p1 = primary_lane_ids[0];
    int p2 = primary_lane_ids[1];
    if (p1 > p2) {
        int temp = p1;
        p1 = p2;
        p2 = temp;
    }

    // Primary가 2개인 경우 외분 생성
    if (primary_count == 2) {
        if (p1 == 1 && p2 == 2) {
            // Primary = lane1, lane2 → outer = lane3, lane4
            // lane3: 1:1 외분 (lane2 기준, lane1 참조)
            ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
            lane3_polyfit_.id = "lane3";
            lane3_is_generated_ = true;

            // lane4: 2:1 외분 (lane2 기준, lane1 참조)
            ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
            lane4_polyfit_.id = "lane4";
            lane4_is_generated_ = true;

        } else if (p1 == 2 && p2 == 3) {
            // Primary = lane2, lane3 → outer = lane1, lane4
            // lane1: 1:1 외분 (lane2 기준, lane3 참조)
            ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
            lane1_polyfit_.id = "lane1";
            lane1_is_generated_ = true;

            // lane4: 1:1 외분 (lane3 기준, lane2 참조)
            ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
            lane4_polyfit_.id = "lane4";
            lane4_is_generated_ = true;

        } else if (p1 == 3 && p2 == 4) {
            // Primary = lane3, lane4 → outer = lane1, lane2
            // lane2: 1:1 외분 (lane3 기준, lane4 참조)
            ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
            lane2_polyfit_.id = "lane2";
            lane2_is_generated_ = true;

            // lane1: 2:1 외분 (lane3 기준, lane4 참조)
            ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
            lane1_polyfit_.id = "lane1";
            lane1_is_generated_ = true;

        } else if (p1 == 1 && p2 == 3) {
            // Edge case: Primary = lane1, lane3 (비인접)
            // lane2: lane1과 lane3의 평균 (내분)
            lane2_polyfit_.a0 = (lane1_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
            lane2_polyfit_.a1 = (lane1_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
            lane2_polyfit_.a2 = (lane1_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
            lane2_polyfit_.a3 = (lane1_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
            lane2_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane3_polyfit_.x_start);
            lane2_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane3_polyfit_.x_end);
            lane2_polyfit_.id = "lane2";
            lane2_is_generated_ = true;

            // lane4: 1:1 외분 (lane3 기준, lane1 참조)
            ExtrapolateLane(lane3_polyfit_, lane1_polyfit_, lane4_polyfit_);
            lane4_polyfit_.id = "lane4";
            lane4_is_generated_ = true;

        } else if (p1 == 2 && p2 == 4) {
            // Edge case: Primary = lane2, lane4 (비인접)
            // lane1: 1:1 외분 (lane2 기준, lane4 참조)
            ExtrapolateLane(lane2_polyfit_, lane4_polyfit_, lane1_polyfit_);
            lane1_polyfit_.id = "lane1";
            lane1_is_generated_ = true;

            // lane3: lane2와 lane4의 평균 (내분)
            lane3_polyfit_.a0 = (lane2_polyfit_.a0 + lane4_polyfit_.a0) / 2.0;
            lane3_polyfit_.a1 = (lane2_polyfit_.a1 + lane4_polyfit_.a1) / 2.0;
            lane3_polyfit_.a2 = (lane2_polyfit_.a2 + lane4_polyfit_.a2) / 2.0;
            lane3_polyfit_.a3 = (lane2_polyfit_.a3 + lane4_polyfit_.a3) / 2.0;
            lane3_polyfit_.x_start = std::max(lane2_polyfit_.x_start, lane4_polyfit_.x_start);
            lane3_polyfit_.x_end = std::min(lane2_polyfit_.x_end, lane4_polyfit_.x_end);
            lane3_polyfit_.id = "lane3";
            lane3_is_generated_ = true;

        } else if (p1 == 1 && p2 == 4) {
            // Edge case: Primary = lane1, lane4 (가장 먼 조합)
            // lane2, lane3: 1/3, 2/3 내분
            lane2_polyfit_.a0 = lane1_polyfit_.a0 * 2.0/3.0 + lane4_polyfit_.a0 * 1.0/3.0;
            lane2_polyfit_.a1 = lane1_polyfit_.a1 * 2.0/3.0 + lane4_polyfit_.a1 * 1.0/3.0;
            lane2_polyfit_.a2 = lane1_polyfit_.a2 * 2.0/3.0 + lane4_polyfit_.a2 * 1.0/3.0;
            lane2_polyfit_.a3 = lane1_polyfit_.a3 * 2.0/3.0 + lane4_polyfit_.a3 * 1.0/3.0;
            lane2_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane4_polyfit_.x_start);
            lane2_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane4_polyfit_.x_end);
            lane2_polyfit_.id = "lane2";
            lane2_is_generated_ = true;

            lane3_polyfit_.a0 = lane1_polyfit_.a0 * 1.0/3.0 + lane4_polyfit_.a0 * 2.0/3.0;
            lane3_polyfit_.a1 = lane1_polyfit_.a1 * 1.0/3.0 + lane4_polyfit_.a1 * 2.0/3.0;
            lane3_polyfit_.a2 = lane1_polyfit_.a2 * 1.0/3.0 + lane4_polyfit_.a2 * 2.0/3.0;
            lane3_polyfit_.a3 = lane1_polyfit_.a3 * 1.0/3.0 + lane4_polyfit_.a3 * 2.0/3.0;
            lane3_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane4_polyfit_.x_start);
            lane3_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane4_polyfit_.x_end);
            lane3_polyfit_.id = "lane3";
            lane3_is_generated_ = true;
        }

    } else if (primary_count == 1) {
        // Fallback: Primary가 1개만 있는 경우
        // 이전 polyfit 유지 또는 단일 primary 기반으로 다른 레인 추정
        // 여기서는 이전 polyfit을 유지하고 generated 플래그만 설정
        if (!lane1_is_primary_) {
            lane1_is_generated_ = true;
        }
        if (!lane2_is_primary_) {
            lane2_is_generated_ = true;
        }
        if (!lane3_is_primary_) {
            lane3_is_generated_ = true;
        }
        if (!lane4_is_primary_) {
            lane4_is_generated_ = true;
        }

    } else {
        // Fallback: Primary가 0개인 경우 (신뢰할 수 있는 레인 없음)
        // 이전 polyfit 유지
        lane1_is_generated_ = true;
        lane2_is_generated_ = true;
        lane3_is_generated_ = true;
        lane4_is_generated_ = true;
    }

    // -------------------------------------------------------------------------
    // Step 6: Driveway 판단 (피팅 완료 후)
    //         ego_center_lane 생성을 위해서만 사용
    // -------------------------------------------------------------------------
    DetermineCurrentDriveway();

    // -------------------------------------------------------------------------
    // Step 7: Lane ordering 체크 (warning만 출력)
    // -------------------------------------------------------------------------
    EnforceLaneOrdering();

    // -------------------------------------------------------------------------
    // Step 8: Ego center lane 생성
    // -------------------------------------------------------------------------
    GenerateEgoCenterLane();
}

// ============================================================================
// Previous Polyfit Storage (롤백용)
// ============================================================================

/**
 * @brief Saves current polyfits to previous polyfit buffers
 */
void LaneProcessor::SavePreviousPolyfits() {
    prev_lane1_polyfit_ = lane1_polyfit_;
    prev_lane2_polyfit_ = lane2_polyfit_;
    prev_lane3_polyfit_ = lane3_polyfit_;
    prev_lane4_polyfit_ = lane4_polyfit_;
}

// ============================================================================
// Lane Ordering Enforcement
// ============================================================================

/**
 * @brief Checks lane ordering constraint (y1 > y2 > y3 > y4)
 *
 * @return true if ordering is correct, false otherwise
 *
 * @details
 * Lane boundary ordering 불변성: y1 > y2 > y3 > y4
 * (lane1이 가장 왼쪽, lane4가 가장 오른쪽)
 *
 * @note 현재는 warning만 출력하고 롤백하지 않음
 */
bool LaneProcessor::EnforceLaneOrdering() {
    const double y1_at_origin = lane1_polyfit_.a0;
    const double y2_at_origin = lane2_polyfit_.a0;
    const double y3_at_origin = lane3_polyfit_.a0;
    const double y4_at_origin = lane4_polyfit_.a0;

    const bool lane1_left_of_lane2 = (y1_at_origin > y2_at_origin);
    const bool lane2_left_of_lane3 = (y2_at_origin > y3_at_origin);
    const bool lane3_left_of_lane4 = (y3_at_origin > y4_at_origin);

    const bool ordering_is_correct =
        lane1_left_of_lane2 && lane2_left_of_lane3 && lane3_left_of_lane4;

    return ordering_is_correct;
}

// ============================================================================
// Error Detection (soft / fatal 분리 + 히스테리시스)
// ============================================================================

/**
 * @brief Detects lane detection errors and determines if HardReset is needed
 *
 * @return true if HardReset should be triggered, false otherwise
 *
 * @details Error categories:
 *   1. NaN/Inf coefficients -> fatal error (즉시 리셋)
 *   2. Lane ordering violation (y1>y2>y3>y4) -> soft error
 *   3. Abnormal lane width -> soft or fatal depending on severity
 *   4. Adjacent lane overlap -> fatal error
 *
 * Reset trigger conditions:
 *   - Fatal error 1프레임 이상 -> 즉시 reset
 *   - Soft error 5프레임 연속 -> reset
 *
 * Cooldown: HARD_RESET_COOLDOWN_SEC(=1.0초) 이내 재리셋 방지
 */
bool LaneProcessor::HasLaneDetectionError() {
    // HardReset active 상태에서는 추가 HardReset 금지
    if (hard_reset_active_) {
        return false;
    }

    bool soft_error_detected = false;
    bool fatal_error_detected = false;

    // -------------------------------------------------------------------------
    // Error Check 1: NaN/Inf 체크 -> fatal 에러
    //                Polyfit 계수 중 하나라도 유한하지 않으면 치명적 에러
    // -------------------------------------------------------------------------
    const bool lane1_has_finite_coeffs = CheckFinitePolyfit(lane1_polyfit_);
    const bool lane2_has_finite_coeffs = CheckFinitePolyfit(lane2_polyfit_);
    const bool lane3_has_finite_coeffs = CheckFinitePolyfit(lane3_polyfit_);
    const bool lane4_has_finite_coeffs = CheckFinitePolyfit(lane4_polyfit_);

    const bool all_polyfits_are_finite =
        lane1_has_finite_coeffs && lane2_has_finite_coeffs &&
        lane3_has_finite_coeffs && lane4_has_finite_coeffs;

    if (!all_polyfits_are_finite) {
        fatal_error_detected = true;
    }

    // -------------------------------------------------------------------------
    // Error Check 2: 레인 순서 체크 (y1 > y2 > y3 > y4) -> soft 에러
    // -------------------------------------------------------------------------
    const double y1_at_origin = lane1_polyfit_.a0;
    const double y2_at_origin = lane2_polyfit_.a0;
    const double y3_at_origin = lane3_polyfit_.a0;
    const double y4_at_origin = lane4_polyfit_.a0;

    const bool lane1_left_of_lane2 = (y1_at_origin > y2_at_origin);
    const bool lane2_left_of_lane3 = (y2_at_origin > y3_at_origin);
    const bool lane3_left_of_lane4 = (y3_at_origin > y4_at_origin);
    const bool ordering_is_correct =
        lane1_left_of_lane2 && lane2_left_of_lane3 && lane3_left_of_lane4;

    if (!ordering_is_correct) {
        soft_error_detected = true;
    }

    // -------------------------------------------------------------------------
    // Error Check 3: 현재 driveway 기준 레인 폭 체크
    //                폭이 MIN_LANE_WIDTH(1.5m)~MAX_LANE_WIDTH(6.0m) 범위 밖이면 에러
    // -------------------------------------------------------------------------
    // 차량 앞쪽 5m 지점에서 레인 폭 평가
    static constexpr double DRIVEWAY_EVALUATION_X = 5.0;

    const interface::PolyfitLane* left_boundary_lane = nullptr;
    const interface::PolyfitLane* right_boundary_lane = nullptr;

    if (current_driveway_ == 1) {
        left_boundary_lane = &lane1_polyfit_;
        right_boundary_lane = &lane2_polyfit_;
    } else if (current_driveway_ == 2) {
        left_boundary_lane = &lane2_polyfit_;
        right_boundary_lane = &lane3_polyfit_;
    } else if (current_driveway_ == 3) {
        left_boundary_lane = &lane3_polyfit_;
        right_boundary_lane = &lane4_polyfit_;
    }

    if (left_boundary_lane != nullptr && right_boundary_lane != nullptr) {
        const double y_left =
            EvaluatePolynomial(*left_boundary_lane, DRIVEWAY_EVALUATION_X);
        const double y_right =
            EvaluatePolynomial(*right_boundary_lane, DRIVEWAY_EVALUATION_X);
        const double driveway_width = std::fabs(y_left - y_right);

        const bool width_too_narrow = (driveway_width < MIN_LANE_WIDTH);
        const bool width_too_wide = (driveway_width > MAX_LANE_WIDTH);

        if (width_too_narrow || width_too_wide) {
            // 극단적인 경우: fatal error
            const bool extremely_narrow = (driveway_width < MIN_LANE_WIDTH * 0.5);
            const bool extremely_wide = (driveway_width > MAX_LANE_WIDTH * 1.5);

            if (extremely_narrow || extremely_wide) {
                fatal_error_detected = true;
            } else {
                soft_error_detected = true;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Error Check 4: 인접 레인 겹침(overlap) 체크 -> fatal 에러
    //                두 인접 레인이 5cm 이내로 가까우면 겹침으로 판단
    // -------------------------------------------------------------------------
    const bool lane1_lane2_overlap = AreLanesTooClose(lane1_polyfit_, lane2_polyfit_);
    const bool lane2_lane3_overlap = AreLanesTooClose(lane2_polyfit_, lane3_polyfit_);
    const bool lane3_lane4_overlap = AreLanesTooClose(lane3_polyfit_, lane4_polyfit_);

    const bool any_lanes_overlap =
        lane1_lane2_overlap || lane2_lane3_overlap || lane3_lane4_overlap;

    if (any_lanes_overlap) {
        fatal_error_detected = true;
    }

    // -------------------------------------------------------------------------
    // Step 5: 에러 카운터 갱신 (히스테리시스)
    // -------------------------------------------------------------------------
    if (fatal_error_detected) {
        consecutive_fatal_error_frames_++;
    } else {
        consecutive_fatal_error_frames_ = 0;
    }

    if (soft_error_detected) {
        consecutive_soft_error_frames_++;
    } else {
        consecutive_soft_error_frames_ = 0;
    }

    // -------------------------------------------------------------------------
    // Step 6: HardReset 트리거 결정
    //         - Fatal error 1프레임 이상 -> 즉시 reset 요청
    //         - Soft error 5프레임 연속 -> reset 요청
    // -------------------------------------------------------------------------
    bool should_trigger_reset = false;

    const bool fatal_threshold_exceeded = (consecutive_fatal_error_frames_ >= 1);
    const bool soft_threshold_exceeded = (consecutive_soft_error_frames_ >= 5);

    if (fatal_threshold_exceeded) {
        should_trigger_reset = true;
    } else if (soft_threshold_exceeded) {
        should_trigger_reset = true;
    }

    if (!should_trigger_reset) {
        return false;
    }

    // -------------------------------------------------------------------------
    // Step 7: HardReset 쿨다운 검사
    //         마지막 리셋 이후 HARD_RESET_COOLDOWN_SEC 이내면 리셋 억제
    // -------------------------------------------------------------------------
    const auto now = std::chrono::steady_clock::now();
    const double current_time_seconds =
        std::chrono::duration<double>(now.time_since_epoch()).count();
    const double elapsed_since_last_reset =
        current_time_seconds - last_hard_reset_time_;

    const bool within_cooldown_period =
        (elapsed_since_last_reset < HARD_RESET_COOLDOWN_SEC);

    if (within_cooldown_period) {
        std::cout << "[LaneProcessor] HardReset suppressed by cooldown: "
                  << "elapsed=" << elapsed_since_last_reset << "s, "
                  << "cooldown=" << HARD_RESET_COOLDOWN_SEC << "s"
                  << std::endl;
        return false;
    }

    // 쿨다운 통과: HardReset 허용, 마지막 리셋 시각 갱신
    last_hard_reset_time_ = current_time_seconds;
    return true;
}

// ============================================================================
// HardReset: 전체 메모리 초기화 및 RECOVERY 모드 진입
// ============================================================================

/**
 * @brief Critical recovery: clear all lane memories and enter RECOVERY mode
 *
 * @details This function performs a complete reset of the lane processor state:
 *   - Clears all lane point buffers (lane1~4, recovery buffers)
 *   - Resets polyfit coefficients to default values
 *   - Resets error counters and frame counters
 *   - Sets tracking_state_ to RECOVERY
 *
 * @note last_hard_reset_time_ is NOT modified here.
 *       Cooldown timing is managed by HasLaneDetectionError().
 */
void LaneProcessor::HardReset() {
    std::cout << "[LaneProcessor] === HardReset called ===" << std::endl;

    // HardReset 락 활성화 (NORMAL 복귀 전까지 재진입 방지)
    hard_reset_active_ = true;

    // -------------------------------------------------------------------------
    // Step 1: Lane 포인트 메모리 초기화
    // -------------------------------------------------------------------------
    lane1_points_.clear();
    lane2_points_.clear();
    lane3_points_.clear();
    lane4_points_.clear();

    prev_lane1_points_.clear();
    prev_lane2_points_.clear();
    prev_lane3_points_.clear();
    prev_lane4_points_.clear();

    // 샘플 포인트 메모리 초기화
    lane1_sample_points_.clear();
    lane2_sample_points_.clear();
    lane3_sample_points_.clear();
    lane4_sample_points_.clear();

    // -------------------------------------------------------------------------
    // Step 2: Polyfit 초기화 (기본 생성자로 리셋)
    // -------------------------------------------------------------------------
    lane1_polyfit_ = interface::PolyfitLane();
    lane2_polyfit_ = interface::PolyfitLane();
    lane3_polyfit_ = interface::PolyfitLane();
    lane4_polyfit_ = interface::PolyfitLane();
    ego_center_lane_ = interface::PolyfitLane();

    lane1_polyfit_.id = "lane1";
    lane2_polyfit_.id = "lane2";
    lane3_polyfit_.id = "lane3";
    lane4_polyfit_.id = "lane4";
    ego_center_lane_.id = "ego_center";

    // -------------------------------------------------------------------------
    // Step 3: Generated 플래그 초기화
    // -------------------------------------------------------------------------
    lane1_is_generated_ = false;
    lane2_is_generated_ = false;
    lane3_is_generated_ = false;
    lane4_is_generated_ = false;

    // -------------------------------------------------------------------------
    // Step 3-1: Primary 플래그 초기화
    // -------------------------------------------------------------------------
    lane1_is_primary_ = false;
    lane2_is_primary_ = false;
    lane3_is_primary_ = false;
    lane4_is_primary_ = false;

    // -------------------------------------------------------------------------
    // Step 4: Driveway 초기화 (기본값 2로 리셋)
    // -------------------------------------------------------------------------
    current_driveway_ = 2;
    prev_driveway_ = 2;

    // -------------------------------------------------------------------------
    // Step 5: RECOVERY 관련 멤버 초기화
    // -------------------------------------------------------------------------
    virtual_left_lane_ = interface::PolyfitLane();
    virtual_right_lane_ = interface::PolyfitLane();
    virtual_left_valid_ = false;
    virtual_right_valid_ = false;
    recovery_frames_ = 0;
    recovery_stable_count_ = 0;

    recovery_left_points_.clear();
    recovery_right_points_.clear();

    // -------------------------------------------------------------------------
    // Step 6: 에러 카운터 초기화
    // -------------------------------------------------------------------------
    consecutive_soft_error_frames_ = 0;
    consecutive_fatal_error_frames_ = 0;

    // -------------------------------------------------------------------------
    // Step 7: 프레임 카운터 초기화
    // -------------------------------------------------------------------------
    total_processed_frames_ = 0;

    // -------------------------------------------------------------------------
    // Step 8: 상태 전환 (NORMAL -> RECOVERY)
    // -------------------------------------------------------------------------
    tracking_state_ = LaneTrackingState::RECOVERY;
}

// ============================================================================
// Driveway Determination (ego 차량이 어느 차선에 있는지 판단)
// ============================================================================

/**
 * @brief Determines which driveway the ego vehicle is currently in
 *
 * @details
 * Driveway 판단 기준:
 *   1. 각 driveway(1~3)의 중심선 Y값 계산 (기준 X = 5m 전방)
 *   2. ego 차량(Y=0)에 가장 가까운 중심선을 가진 driveway 선택
 *   3. 단, 폭이 비정상(MIN_LANE_WIDTH~MAX_LANE_WIDTH 범위 밖)인 driveway는 제외
 *
 * 유효한 후보가 없으면 current_driveway_ 유지
 */
void LaneProcessor::DetermineCurrentDriveway() {
    // 기준 x 위치: 차량 앞쪽 5m 지점에서 레인 구조 평가
    static constexpr double DRIVEWAY_EVALUATION_X = 5.0;
    // 드라이브웨이 스위칭 안정 프레임 수 (히스테리시스)
    static constexpr int DRIVEWAY_SWITCH_STABLE_FRAMES = 5;

    int best_driveway_id = current_driveway_;
    double best_center_distance = std::numeric_limits<double>::max();
    bool has_valid_candidate = false;

    // -------------------------------------------------------------------------
    // 각 driveway 후보 평가
    //   - Driveway 1: lane1-lane2 사이
    //   - Driveway 2: lane2-lane3 사이
    //   - Driveway 3: lane3-lane4 사이
    // -------------------------------------------------------------------------
    const bool driveway1_valid = TryAddDrivewayCandidate(
        1, lane1_polyfit_, lane2_polyfit_,
        DRIVEWAY_EVALUATION_X, best_driveway_id, best_center_distance);

    if (driveway1_valid) {
        has_valid_candidate = true;
    }

    const bool driveway2_valid = TryAddDrivewayCandidate(
        2, lane2_polyfit_, lane3_polyfit_,
        DRIVEWAY_EVALUATION_X, best_driveway_id, best_center_distance);

    if (driveway2_valid) {
        has_valid_candidate = true;
    }

    const bool driveway3_valid = TryAddDrivewayCandidate(
        3, lane3_polyfit_, lane4_polyfit_,
        DRIVEWAY_EVALUATION_X, best_driveway_id, best_center_distance);

    if (driveway3_valid) {
        has_valid_candidate = true;
    }

    // 유효한 후보가 하나도 없는 경우: current_driveway_ 유지
    if (!has_valid_candidate) {
        return;
    }

    // -------------------------------------------------------------------------
    // 히스테리시스 적용: 여러 프레임 연속 후보가 유지될 때만 실제 변경
    // -------------------------------------------------------------------------

    // 1) best_driveway_id가 현재 driveway와 같으면, 변경 후보 초기화
    if (best_driveway_id == current_driveway_) {
        driveway_change_candidate_id_ = current_driveway_;
        driveway_change_candidate_count_ = 0;
        return;
    }

    // 2) best_driveway_id가 현재 후보와 같으면 카운트 증가,
    //    다르면 새로운 후보로 리셋
    if (best_driveway_id == driveway_change_candidate_id_) {
        driveway_change_candidate_count_++;
    } else {
        driveway_change_candidate_id_ = best_driveway_id;
        driveway_change_candidate_count_ = 1;
    }

    // 3) 같은 후보가 DRIVEWAY_SWITCH_STABLE_FRAMES 프레임 연속 나오면 실제로 변경
    if (driveway_change_candidate_count_ >= DRIVEWAY_SWITCH_STABLE_FRAMES) {
        prev_driveway_ = current_driveway_;
        current_driveway_ = driveway_change_candidate_id_;
        driveway_change_candidate_count_ = 0;
    }
}

/**
 * @brief Evaluates a driveway candidate and updates best if closer to ego
 *
 * @param driveway_id ID of the driveway (1, 2, or 3)
 * @param left_lane Left boundary lane of the driveway
 * @param right_lane Right boundary lane of the driveway
 * @param evaluation_x X position for evaluating lane Y values
 * @param[in,out] out_best_id Current best driveway ID (updated if this is better)
 * @param[in,out] out_best_center_distance Current best center distance (updated)
 * @return true if this driveway is a valid candidate (width in range)
 */
bool LaneProcessor::TryAddDrivewayCandidate(
    int driveway_id,
    const interface::PolyfitLane& left_lane,
    const interface::PolyfitLane& right_lane,
    double evaluation_x,
    int& out_best_id,
    double& out_best_center_distance) const {

    // 해당 X에서 좌/우 경계 Y값 계산
    const double y_left = EvaluatePolynomial(left_lane, evaluation_x);
    const double y_right = EvaluatePolynomial(right_lane, evaluation_x);
    const double driveway_width = std::fabs(y_left - y_right);

    // 폭이 비정상적인 driveway는 후보에서 제외
    const bool width_too_narrow = (driveway_width < MIN_LANE_WIDTH);
    const bool width_too_wide = (driveway_width > MAX_LANE_WIDTH);

    if (width_too_narrow || width_too_wide) {
        return false;
    }

    // Driveway 중심선 Y값 계산
    const double center_y = 0.5 * (y_left + y_right);

    // ego(Y=0)에서 중심선까지 거리
    const double distance_to_ego = std::fabs(center_y);

    // ego에 더 가까운 driveway가 있으면 업데이트
    if (distance_to_ego < out_best_center_distance) {
        out_best_center_distance = distance_to_ego;
        out_best_id = driveway_id;
    }

    return true;
}

// ============================================================================
// Ego Center Lane Generation
// ============================================================================

/**
 * @brief Generates ego center lane from current driveway boundaries
 *
 * @details
 * Ego center lane = 현재 driveway의 좌/우 경계 평균
 *   - Driveway 1: (lane1 + lane2) / 2
 *   - Driveway 2: (lane2 + lane3) / 2
 *   - Driveway 3: (lane3 + lane4) / 2
 *
 * X 범위는 두 경계의 교집합 사용
 */
void LaneProcessor::GenerateEgoCenterLane() {
    ego_center_lane_.id = "ego_center";
    ego_center_lane_.frame_id = "vehicle";

    if (current_driveway_ == 1) {
        // Driveway 1: (lane1 + lane2) / 2
        ego_center_lane_.a0 = (lane1_polyfit_.a0 + lane2_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane1_polyfit_.a1 + lane2_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane1_polyfit_.a2 + lane2_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane1_polyfit_.a3 + lane2_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane1_polyfit_.x_start, lane2_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane1_polyfit_.x_end, lane2_polyfit_.x_end);

    } else if (current_driveway_ == 2) {
        // Driveway 2: (lane2 + lane3) / 2
        ego_center_lane_.a0 = (lane2_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane2_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane2_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane2_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);

    } else if (current_driveway_ == 3) {
        // Driveway 3: (lane3 + lane4) / 2
        ego_center_lane_.a0 = (lane3_polyfit_.a0 + lane4_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane3_polyfit_.a1 + lane4_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane3_polyfit_.a2 + lane4_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane3_polyfit_.a3 + lane4_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane3_polyfit_.x_start, lane4_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane3_polyfit_.x_end, lane4_polyfit_.x_end);
    }
}

// ============================================================================
// Helper Functions - Downsampling
// ============================================================================

/**
 * @brief Downsamples all lane point buffers
 */
void LaneProcessor::DownsampleLanePoints() {
    DownsampleSingleLaneWithTrigger(lane1_points_);
    DownsampleSingleLaneWithTrigger(lane2_points_);
    DownsampleSingleLaneWithTrigger(lane3_points_);
    DownsampleSingleLaneWithTrigger(lane4_points_);
}

/**
 * @brief Downsamples a single lane point buffer (immediate, no trigger)
 *
 * @param[in,out] lane_points Points to downsample
 *
 * @details
 * MAX_POINTS_PER_LANE(=100)개 초과 시 균등 간격 샘플링으로 축소
 * X 좌표 기준 정렬 후 균등 인덱스 선택
 */
void LaneProcessor::DownsampleSingleLane(std::vector<interface::Point2D>& lane_points) {
    const size_t current_size = lane_points.size();
    const bool within_limit = (current_size <= MAX_POINTS_PER_LANE);

    if (within_limit) {
        return;
    }

    // X 좌표 기준 정렬
    std::sort(lane_points.begin(), lane_points.end(), ComparePointsByX);

    // 균등 간격 샘플링
    std::vector<interface::Point2D> downsampled_points;
    downsampled_points.reserve(MAX_POINTS_PER_LANE);

    for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
        const size_t source_index = (i * current_size) / MAX_POINTS_PER_LANE;
        if (source_index < current_size) {
            downsampled_points.push_back(lane_points[source_index]);
        }
    }

    lane_points = std::move(downsampled_points);
}

/**
 * @brief Downsamples a single lane with trigger factor (hysteresis)
 *
 * @param[in,out] lane_points Points to downsample
 *
 * @details
 * MAX_POINTS_PER_LANE의 1.5배(=150개)를 초과할 때만 다운샘플 수행
 * 이를 통해 빈번한 다운샘플링 방지
 */
void LaneProcessor::DownsampleSingleLaneWithTrigger(
    std::vector<interface::Point2D>& lane_points) {

    // 트리거 팩터: 1.5배 초과 시에만 다운샘플
    static constexpr double DOWNSAMPLE_TRIGGER_FACTOR = 1.5;
    const size_t trigger_threshold =
        static_cast<size_t>(MAX_POINTS_PER_LANE * DOWNSAMPLE_TRIGGER_FACTOR);

    const bool below_trigger_threshold = (lane_points.size() <= trigger_threshold);

    if (below_trigger_threshold) {
        return;
    }

    // X 좌표 기준 정렬
    std::sort(lane_points.begin(), lane_points.end(), ComparePointsByX);

    // 균등 간격 샘플링
    std::vector<interface::Point2D> downsampled_points;
    downsampled_points.reserve(MAX_POINTS_PER_LANE);

    const size_t total_points = lane_points.size();

    for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
        const size_t source_index =
            (total_points == 0) ? 0 : (i * total_points) / MAX_POINTS_PER_LANE;
        if (source_index < total_points) {
            downsampled_points.push_back(lane_points[source_index]);
        }
    }

    lane_points = std::move(downsampled_points);
}

// ============================================================================
// Helper Functions - Filtering
// ============================================================================

/**
 * @brief Filters stored points by X and Y range
 *
 * @param vehicle_state Current vehicle state (unused)
 *
 * @details
 * 메모리에서 범위 밖 포인트 제거:
 *   - X: MEMORY_X_MIN(-15m) ~ MEMORY_X_MAX(+20m) 범위
 *   - Y: +/-MAX_LATERAL_MEMORY(6m) 범위
 *
 * @note driveway 변경 시 outer lane 메모리는 RebuildOuterLanesOnDrivewayChange()에서
 *       강제로 clear되므로, 여기서는 장기적인 메모리만 제거
 */
void LaneProcessor::FilterStoredPointsByRange(
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // Unused parameter

    // Y 범위 필터 상수: 도로 폭에 맞게 설정 (차량 좌우 6m)
    static constexpr double MAX_LATERAL_MEMORY = 6.0;

    FilterLanePointsByXYRange(lane1_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane2_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane3_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane4_points_, MAX_LATERAL_MEMORY);
}

/**
 * @brief Filters points by X range only (MEMORY_X_MIN ~ MEMORY_X_MAX)
 *
 * @param[in,out] lane_points Points to filter (modified in place)
 *
 * @details
 * 차량 좌표계 기준:
 *   - X: 전방 양수, 후방 음수
 *   - 유효 범위: -15m ~ +20m
 */
void LaneProcessor::FilterPointsByXRange(std::vector<interface::Point2D>& lane_points) {
    size_t write_index = 0;

    for (size_t read_index = 0; read_index < lane_points.size(); ++read_index) {
        const interface::Point2D& point = lane_points[read_index];

        const bool x_in_range =
            (point.x >= MEMORY_X_MIN) && (point.x <= MEMORY_X_MAX);

        if (x_in_range) {
            if (write_index != read_index) {
                lane_points[write_index] = point;
            }
            ++write_index;
        }
    }

    lane_points.resize(write_index);
}

/**
 * @brief Filters points by both X and Y range
 *
 * @param[in,out] lane_points Points to filter (modified in place)
 * @param max_lateral_distance Maximum absolute Y value (symmetric +/-)
 *
 * @details
 * 차량 좌표계 기준 유효 범위:
 *   - X: MEMORY_X_MIN(-15m) ~ MEMORY_X_MAX(+20m)
 *   - Y: -max_lateral_distance ~ +max_lateral_distance
 */
void LaneProcessor::FilterLanePointsByXYRange(
    std::vector<interface::Point2D>& lane_points,
    double max_lateral_distance) {

    size_t write_index = 0;

    for (size_t read_index = 0; read_index < lane_points.size(); ++read_index) {
        const interface::Point2D& point = lane_points[read_index];

        const bool x_in_range =
            (point.x >= MEMORY_X_MIN) && (point.x <= MEMORY_X_MAX);
        const bool y_in_range =
            (point.y >= -max_lateral_distance) && (point.y <= max_lateral_distance);
        const bool point_in_valid_range = x_in_range && y_in_range;

        if (point_in_valid_range) {
            if (write_index != read_index) {
                lane_points[write_index] = point;
            }
            ++write_index;
        }
    }

    lane_points.resize(write_index);
}

// ============================================================================
// Helper Functions - Polynomial Operations
// ============================================================================

/**
 * @brief Evaluates a cubic polynomial at given x
 *
 * @param polyfit Polynomial coefficients (a0, a1, a2, a3)
 * @param x X coordinate to evaluate at
 * @return y = a0 + a1*x + a2*x^2 + a3*x^3
 */
double LaneProcessor::EvaluatePolynomial(
    const interface::PolyfitLane& polyfit,
    double x) const {

    return polyfit.a0 +
           polyfit.a1 * x +
           polyfit.a2 * x * x +
           polyfit.a3 * x * x * x;
}

/**
 * @brief Fits a cubic polynomial to a set of points using least squares
 *
 * @param points Input points to fit
 * @param[out] out_polyfit Output polynomial coefficients
 * @return true if fitting succeeded (at least 4 points)
 *
 * @details
 * 최소자승법으로 3차 다항식 피팅:
 *   y = a0 + a1*x + a2*x^2 + a3*x^3
 *
 * Vandermonde 행렬 A와 y 벡터 b를 구성하여
 * (A^T * A) * coeffs = A^T * b 를 LDLT 분해로 해결
 */
bool LaneProcessor::FitCubicPolynomial(
    const std::vector<interface::Point2D>& points,
    interface::PolyfitLane& out_polyfit) const {

    // 최소 4개 포인트 필요 (3차 다항식)
    static constexpr size_t MIN_POINTS_FOR_CUBIC_FIT = 4;

    if (points.size() < MIN_POINTS_FOR_CUBIC_FIT) {
        return false;
    }

    const size_t num_points = points.size();

    // Vandermonde 행렬 구성
    Eigen::MatrixXd design_matrix(num_points, 4);
    Eigen::VectorXd y_vector(num_points);

    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < num_points; ++i) {
        const double x = points[i].x;
        const double y = points[i].y;

        design_matrix(i, 0) = 1.0;
        design_matrix(i, 1) = x;
        design_matrix(i, 2) = x * x;
        design_matrix(i, 3) = x * x * x;
        y_vector(i) = y;

        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
    }

    // 정규방정식 해결: (A^T * A) * coeffs = A^T * b
    const Eigen::MatrixXd normal_matrix =
        design_matrix.transpose() * design_matrix;
    const Eigen::VectorXd normal_rhs =
        design_matrix.transpose() * y_vector;
    const Eigen::VectorXd coefficients = normal_matrix.ldlt().solve(normal_rhs);

    // 결과 저장
    out_polyfit.a0 = coefficients(0);
    out_polyfit.a1 = coefficients(1);
    out_polyfit.a2 = coefficients(2);
    out_polyfit.a3 = coefficients(3);
    out_polyfit.x_start = x_min;
    out_polyfit.x_end = x_max;

    return true;
}

// ============================================================================
// Primary Lane Fitting with Memory and Samples
// ============================================================================

/**
 * @brief Fits a primary lane using memory + sample memory + new samples
 *
 * @param[in,out] lane_points Real points buffer (ego-motion compensated)
 * @param[in,out] sample_points Sample points buffer (ego-motion compensated)
 * @param[in,out] lane_polyfit Polynomial to update
 * @param[in] vehicle_state Current vehicle state for coordinate transform
 *
 * @details
 * Primary lane에 대해 다음 데이터를 사용하여 피팅:
 *   1. 실측 메모리 포인트 (lane_points)
 *   2. 샘플 메모리 포인트 (sample_points, ego-motion compensated)
 *   3. 새로 생성한 샘플 (실측+샘플메모리 모두 없는 구간만)
 *
 * 샘플 메모리 관리:
 *   - 실측 포인트가 있는 구간의 샘플은 제거
 *   - 새 샘플 생성 시 샘플 메모리에 저장 (ego-motion compensation 대상)
 *   - MAX_POINTS_PER_LANE 제한 적용
 */
void LaneProcessor::FitPrimaryLaneWithMemoryAndSamples(
    std::vector<interface::Point2D>& lane_points,
    std::vector<interface::Point2D>& sample_points,
    interface::PolyfitLane& lane_polyfit,
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // 샘플은 차량 좌표계로 저장, ego-motion compensation으로 일관성 유지

    // -------------------------------------------------------------------------
    // Step 1: 실측 포인트가 비어 있으면 아무 것도 하지 않음
    // -------------------------------------------------------------------------
    if (lane_points.empty()) {
        return;
    }

    // -------------------------------------------------------------------------
    // Step 2: 샘플 메모리에서 실측 포인트 구간 제거
    //         실측 포인트 근처(GAP_THRESHOLD)에 있는 샘플은 삭제
    // -------------------------------------------------------------------------
    FilterSamplePointsByRealPoints(sample_points, lane_points);

    // -------------------------------------------------------------------------
    // Step 3: 이전 polyfit의 x 범위 결정
    // -------------------------------------------------------------------------
    double sample_x_start = lane_polyfit.x_start;
    double sample_x_end = lane_polyfit.x_end;

    const bool invalid_x_range = (sample_x_end <= sample_x_start);
    if (invalid_x_range) {
        sample_x_start = MEMORY_X_MIN;
        sample_x_end = MEMORY_X_MAX;
    }

    // -------------------------------------------------------------------------
    // Step 4: 새 샘플 포인트 생성 (실측 + 샘플메모리 모두 없는 구간만)
    // -------------------------------------------------------------------------
    std::vector<interface::Point2D> new_samples;
    new_samples.reserve(PRIMARY_LANE_NUM_SAMPLES);

    for (int i = 0; i < PRIMARY_LANE_NUM_SAMPLES; ++i) {
        const double ratio = (PRIMARY_LANE_NUM_SAMPLES <= 1)
            ? 0.0
            : static_cast<double>(i) / static_cast<double>(PRIMARY_LANE_NUM_SAMPLES - 1);

        const double sample_x = sample_x_start + ratio * (sample_x_end - sample_x_start);

        // 실측 포인트 근처인지 검사
        bool has_nearby_real = false;
        for (const auto& pt : lane_points) {
            if (std::fabs(pt.x - sample_x) < PRIMARY_LANE_GAP_THRESHOLD) {
                has_nearby_real = true;
                break;
            }
        }

        // 샘플 메모리 근처인지 검사
        bool has_nearby_sample = false;
        for (const auto& pt : sample_points) {
            if (std::fabs(pt.x - sample_x) < PRIMARY_LANE_GAP_THRESHOLD) {
                has_nearby_sample = true;
                break;
            }
        }

        // 실측/샘플 모두 없는 구간에만 새 샘플 생성
        if (!has_nearby_real && !has_nearby_sample) {
            const double sample_y = EvaluatePolynomial(lane_polyfit, sample_x);

            if (std::isfinite(sample_y)) {
                interface::Point2D new_sample;
                new_sample.x = sample_x;
                new_sample.y = sample_y;
                new_samples.push_back(new_sample);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: 새 샘플을 샘플 메모리에 저장 (차량 좌표계 그대로)
    //         ego-motion compensation이 적용되므로 다음 프레임에도 유효
    // -------------------------------------------------------------------------
    for (const auto& pt : new_samples) {
        sample_points.push_back(pt);
    }

    // 샘플 메모리 크기 제한 적용
    if (sample_points.size() > MAX_POINTS_PER_LANE) {
        DownsampleSingleLane(sample_points);
    }

    // X 범위 필터링 (너무 뒤로 간 샘플 제거)
    FilterPointsByXRange(sample_points);

    // -------------------------------------------------------------------------
    // Step 6: combined_points 구성 (실측 + 샘플 메모리)
    // -------------------------------------------------------------------------
    std::vector<interface::Point2D> combined_points;
    combined_points.reserve(lane_points.size() + sample_points.size());

    for (const auto& pt : lane_points) {
        combined_points.push_back(pt);
    }

    for (const auto& pt : sample_points) {
        combined_points.push_back(pt);
    }

    // -------------------------------------------------------------------------
    // Step 7: 피팅 수행
    // -------------------------------------------------------------------------
    FitCubicPolynomial(combined_points, lane_polyfit);
}

/**
 * @brief 샘플 포인트 메모리에서 실측 포인트 구간의 샘플 제거
 *
 * @param[in,out] sample_points 샘플 포인트 메모리 (수정됨)
 * @param[in] real_points 실측 포인트 (참조용)
 *
 * @details
 * 실측 포인트 근처(PRIMARY_LANE_GAP_THRESHOLD 이내)에 있는
 * 샘플 포인트를 제거하여 실측 데이터 우선을 보장
 */
void LaneProcessor::FilterSamplePointsByRealPoints(
    std::vector<interface::Point2D>& sample_points,
    const std::vector<interface::Point2D>& real_points) {

    if (sample_points.empty() || real_points.empty()) {
        return;
    }

    std::vector<interface::Point2D> filtered;
    filtered.reserve(sample_points.size());

    for (const auto& sample : sample_points) {
        bool near_real = false;
        for (const auto& real : real_points) {
            if (std::fabs(sample.x - real.x) < PRIMARY_LANE_GAP_THRESHOLD) {
                near_real = true;
                break;
            }
        }
        if (!near_real) {
            filtered.push_back(sample);
        }
    }

    sample_points = std::move(filtered);
}

/**
 * @brief Fits a single lane with reliability check
 *
 * @param[in,out] lane_points Points to fit (may be used for refit)
 * @param[in,out] lane_polyfit Output polynomial (may be updated)
 * @param[out] has_real_points Whether lane has any measured points
 * @param relax_condition Use relaxed thresholds (for driveway change)
 *
 * @details
 * 피팅 신뢰도 조건:
 *   - 정상: MIN_RELIABLE_POINTS(30)개 이상, MIN_X_RANGE(20m) 이상
 *   - 완화(driveway 변경 시): 10개 이상, 5m 이상
 *
 * 신뢰도가 낮으면 이전 polyfit 샘플과 혼합하여 refit
 */
void LaneProcessor::FitSingleLane(
    std::vector<interface::Point2D>& lane_points,
    interface::PolyfitLane& lane_polyfit,
    bool& has_real_points,
    bool relax_condition) {

    // 신뢰도 판단 기준
    static constexpr size_t MIN_RELIABLE_POINTS = 30;  // 최소 포인트 수
    static constexpr double MIN_X_RANGE = 20.0;        // 최소 X 범위 [m]

    has_real_points = !lane_points.empty();

    // X 범위 계산
    double x_min = 0.0;
    double x_max = 0.0;
    const double x_range = ComputeXRange(lane_points, x_min, x_max);

    // Driveway 변경 시 완화된 기준 적용
    const size_t required_points = relax_condition ? 10 : MIN_RELIABLE_POINTS;
    const double required_x_range = relax_condition ? 5.0 : MIN_X_RANGE;

    // 신뢰도 조건 체크
    const bool has_enough_points = (lane_points.size() >= required_points);
    const bool has_enough_x_range = (x_range >= required_x_range);
    const bool is_reliable = has_real_points && has_enough_points && has_enough_x_range;

    if (is_reliable) {
        // 신뢰도 높음: 현재 포인트만으로 피팅
        FitCubicPolynomial(lane_points, lane_polyfit);
    } else if (has_real_points && !relax_condition) {
        // 신뢰도 낮음(정상 모드): 이전 polyfit 샘플과 혼합하여 refit
        RefitWithSamples(lane_points, lane_polyfit);
    }
    // relax_condition=true && !is_reliable: RefitWithSamples 타지 않음
    // (driveway 변경 직후에는 이전 polyfit이 다른 레인일 수 있음)
}

/**
 * @brief Refits polynomial using current points plus samples from previous polyfit
 *
 * @param points Current measured points
 * @param[in,out] lane_polyfit Polynomial to update
 * @return true if refit succeeded
 *
 * @details
 * 포인트가 부족할 때 안정성을 위해:
 *   1. 이전 polyfit에서 MAX_POINTS_PER_LANE개 샘플 생성
 *   2. 현재 포인트와 합쳐서 재피팅
 *
 * 이를 통해 포인트가 적어도 이전 형상을 어느 정도 유지
 */
bool LaneProcessor::RefitWithSamples(
    const std::vector<interface::Point2D>& points,
    interface::PolyfitLane& lane_polyfit) {

    if (points.empty()) {
        return false;
    }

    // 현재 포인트 복사
    std::vector<interface::Point2D> combined_points = points;

    // 이전 polyfit의 X 범위 결정
    double sample_x_start = lane_polyfit.x_start;
    double sample_x_end = lane_polyfit.x_end;

    const bool invalid_x_range = (sample_x_end <= sample_x_start);
    if (invalid_x_range) {
        sample_x_start = MEMORY_X_MIN;
        sample_x_end = MEMORY_X_MAX;
    }

    // 이전 polyfit에서 샘플 포인트 생성
    for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
        // 균등 간격 비율 계산
        const double ratio = (MAX_POINTS_PER_LANE == 1)
            ? 0.0
            : static_cast<double>(i) / static_cast<double>(MAX_POINTS_PER_LANE - 1);

        const double sample_x = sample_x_start + ratio * (sample_x_end - sample_x_start);
        const double sample_y = EvaluatePolynomial(lane_polyfit, sample_x);

        interface::Point2D sample_point;
        sample_point.x = sample_x;
        sample_point.y = sample_y;
        combined_points.push_back(sample_point);
    }

    // 합쳐진 포인트로 재피팅
    return FitCubicPolynomial(combined_points, lane_polyfit);
}

// ============================================================================
// Helper Functions - Range and Validation
// ============================================================================

/**
 * @brief Computes X range of a point set
 *
 * @param points Input points
 * @param[out] out_x_min Minimum X value found
 * @param[out] out_x_max Maximum X value found
 * @return X range (max - min), or 0 if empty
 */
double LaneProcessor::ComputeXRange(
    const std::vector<interface::Point2D>& points,
    double& out_x_min,
    double& out_x_max) const {

    if (points.empty()) {
        out_x_min = 0.0;
        out_x_max = 0.0;
        return 0.0;
    }

    out_x_min = std::numeric_limits<double>::max();
    out_x_max = std::numeric_limits<double>::lowest();

    for (const auto& point : points) {
        out_x_min = std::min(out_x_min, point.x);
        out_x_max = std::max(out_x_max, point.x);
    }

    return out_x_max - out_x_min;
}

/**
 * @brief Checks if all polynomial coefficients are finite (not NaN or Inf)
 *
 * @param polyfit Polynomial to check
 * @return true if all coefficients (a0, a1, a2, a3) are finite
 */
bool LaneProcessor::CheckFinitePolyfit(const interface::PolyfitLane& polyfit) const {
    const bool a0_finite = std::isfinite(polyfit.a0);
    const bool a1_finite = std::isfinite(polyfit.a1);
    const bool a2_finite = std::isfinite(polyfit.a2);
    const bool a3_finite = std::isfinite(polyfit.a3);

    return a0_finite && a1_finite && a2_finite && a3_finite;
}

/**
 * @brief Checks if two adjacent lanes are too close (overlapping)
 *
 * @param lane_a First lane polynomial
 * @param lane_b Second lane polynomial
 * @return true if any sampled point pair is closer than 5cm
 *
 * @details
 * 겹침 판단 방법:
 *   1. 두 레인의 공통 X 범위 결정 (0 ~ 15m로 제한)
 *   2. 10개 샘플 포인트에서 Y 차이 계산
 *   3. 어느 한 점이라도 5cm 미만이면 겹침으로 판단
 */
bool LaneProcessor::AreLanesTooClose(
    const interface::PolyfitLane& lane_a,
    const interface::PolyfitLane& lane_b) const {

    // 공통 X 범위 계산 (전방 0~15m 범위로 제한)
    const double x_start = std::max({lane_a.x_start, lane_b.x_start, 0.0});
    const double x_end = std::min({lane_a.x_end, lane_b.x_end, 15.0});

    const bool no_overlap_range = (x_end <= x_start);
    if (no_overlap_range) {
        return false;
    }

    // 샘플링 파라미터
    static constexpr int NUM_SAMPLE_POINTS = 10;
    static constexpr double OVERLAP_THRESHOLD = 0.05;  // 5cm 이하면 겹침

    for (int i = 0; i <= NUM_SAMPLE_POINTS; ++i) {
        const double ratio = static_cast<double>(i) / NUM_SAMPLE_POINTS;
        const double sample_x = x_start + ratio * (x_end - x_start);

        const double y_a = EvaluatePolynomial(lane_a, sample_x);
        const double y_b = EvaluatePolynomial(lane_b, sample_x);
        const double y_difference = std::fabs(y_a - y_b);

        // NaN 체크
        const bool difference_is_valid = std::isfinite(y_difference);
        if (!difference_is_valid) {
            continue;
        }

        // 겹침 판단
        const bool lanes_too_close = (y_difference < OVERLAP_THRESHOLD);
        if (lanes_too_close) {
            return true;
        }
    }

    return false;
}

// ============================================================================
// Helper Functions - Coordinate Transformation
// ============================================================================

/**
 * @brief Transforms a point from vehicle frame to global frame
 *
 * @param vehicle_point Point in vehicle frame (X-forward, Y-left)
 * @param vehicle_state Vehicle pose in global frame
 * @return Point in global frame
 *
 * @details
 * 변환 공식:
 *   global_x = veh_state.x + cos(yaw)*veh_x - sin(yaw)*veh_y
 *   global_y = veh_state.y + sin(yaw)*veh_x + cos(yaw)*veh_y
 */
interface::Point2D LaneProcessor::VehicleToGlobalFrame(
    const interface::Point2D& vehicle_point,
    const interface::VehicleState& vehicle_state) const {

    const double cos_yaw = std::cos(vehicle_state.yaw);
    const double sin_yaw = std::sin(vehicle_state.yaw);

    interface::Point2D global_point;
    global_point.x =
        vehicle_state.x + cos_yaw * vehicle_point.x - sin_yaw * vehicle_point.y;
    global_point.y =
        vehicle_state.y + sin_yaw * vehicle_point.x + cos_yaw * vehicle_point.y;

    return global_point;
}

/**
 * @brief Transforms a point from global frame to vehicle frame
 *
 * @param global_point Point in global frame
 * @param vehicle_state Vehicle pose in global frame
 * @return Point in vehicle frame (X-forward, Y-left)
 *
 * @details
 * 변환 공식 (역변환):
 *   dx = global_x - veh_state.x
 *   dy = global_y - veh_state.y
 *   veh_x = cos(-yaw)*dx - sin(-yaw)*dy
 *   veh_y = sin(-yaw)*dx + cos(-yaw)*dy
 */
interface::Point2D LaneProcessor::GlobalToVehicleFrame(
    const interface::Point2D& global_point,
    const interface::VehicleState& vehicle_state) const {

    const double delta_x = global_point.x - vehicle_state.x;
    const double delta_y = global_point.y - vehicle_state.y;

    const double cos_neg_yaw = std::cos(-vehicle_state.yaw);
    const double sin_neg_yaw = std::sin(-vehicle_state.yaw);

    interface::Point2D vehicle_point;
    vehicle_point.x = cos_neg_yaw * delta_x - sin_neg_yaw * delta_y;
    vehicle_point.y = sin_neg_yaw * delta_x + cos_neg_yaw * delta_y;

    return vehicle_point;
}

// ============================================================================
// Getter Functions (Public Interface)
// ============================================================================

/**
 * @brief Returns lane1 point buffer (leftmost lane)
 */
const std::vector<interface::Point2D>& LaneProcessor::GetLane1Points() const {
    return lane1_points_;
}

/**
 * @brief Returns lane2 point buffer
 */
const std::vector<interface::Point2D>& LaneProcessor::GetLane2Points() const {
    return lane2_points_;
}

/**
 * @brief Returns lane3 point buffer
 */
const std::vector<interface::Point2D>& LaneProcessor::GetLane3Points() const {
    return lane3_points_;
}

/**
 * @brief Returns lane4 point buffer (rightmost lane)
 */
const std::vector<interface::Point2D>& LaneProcessor::GetLane4Points() const {
    return lane4_points_;
}

/**
 * @brief Returns lane1 polynomial fit
 */
const interface::PolyfitLane& LaneProcessor::GetLane1Polyfit() const {
    return lane1_polyfit_;
}

/**
 * @brief Returns lane2 polynomial fit
 */
const interface::PolyfitLane& LaneProcessor::GetLane2Polyfit() const {
    return lane2_polyfit_;
}

/**
 * @brief Returns lane3 polynomial fit
 */
const interface::PolyfitLane& LaneProcessor::GetLane3Polyfit() const {
    return lane3_polyfit_;
}

/**
 * @brief Returns lane4 polynomial fit
 */
const interface::PolyfitLane& LaneProcessor::GetLane4Polyfit() const {
    return lane4_polyfit_;
}

/**
 * @brief Returns ego center lane polynomial
 */
const interface::PolyfitLane& LaneProcessor::GetEgoLane() const {
    return ego_center_lane_;
}

/**
 * @brief Returns current driveway ID (1, 2, or 3)
 */
int LaneProcessor::GetCurrentDriveway() const {
    return current_driveway_;
}

/**
 * @brief Returns count of valid lane clusters
 *
 * A cluster is considered valid if it has at least RELIABLE_CLUSTER_MIN_POINTS points
 */
int LaneProcessor::GetValidClusterCount() const {
    int count = 0;
    if (lane1_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS) {
        count++;
    }
    if (lane2_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS) {
        count++;
    }
    if (lane3_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS) {
        count++;
    }
    if (lane4_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS) {
        count++;
    }
    return count;
}

/**
 * @brief Returns whether lane1 has enough points to be valid
 */
bool LaneProcessor::IsLane1Valid() const {
    return lane1_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

/**
 * @brief Returns whether lane2 has enough points to be valid
 */
bool LaneProcessor::IsLane2Valid() const {
    return lane2_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

/**
 * @brief Returns whether lane3 has enough points to be valid
 */
bool LaneProcessor::IsLane3Valid() const {
    return lane3_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

/**
 * @brief Returns whether lane4 has enough points to be valid
 */
bool LaneProcessor::IsLane4Valid() const {
    return lane4_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

// ============================================================================
// ROS Message Conversion (visualization용)
// ============================================================================

/**
 * @brief Converts lane1 points to ROS geometry_msgs::Point format
 *
 * @param vehicle_state Current vehicle state (unused)
 * @return Vector of ROS Point messages in vehicle frame
 */
std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane1PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;  // Unused parameter

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane1_points_.size());

    for (const auto& vehicle_pt : lane1_points_) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = vehicle_pt.x;
        ros_point.y = vehicle_pt.y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

/**
 * @brief Converts lane2 points to ROS geometry_msgs::Point format
 */
std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane2PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane2_points_.size());

    for (const auto& vehicle_pt : lane2_points_) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = vehicle_pt.x;
        ros_point.y = vehicle_pt.y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

/**
 * @brief Converts lane3 points to ROS geometry_msgs::Point format
 */
std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane3PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane3_points_.size());

    for (const auto& vehicle_pt : lane3_points_) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = vehicle_pt.x;
        ros_point.y = vehicle_pt.y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

/**
 * @brief Converts lane4 points to ROS geometry_msgs::Point format
 */
std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane4PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane4_points_.size());

    for (const auto& vehicle_pt : lane4_points_) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = vehicle_pt.x;
        ros_point.y = vehicle_pt.y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

}  // namespace lane_processor
