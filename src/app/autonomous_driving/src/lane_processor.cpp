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
 *   - Global/Vehicle frame memory management
 *   - Hard reset and recovery logic for handling lane detection errors
 *
 * The module maintains a state machine with three states:
 *   - INIT: Initial frames where lane memory is being populated
 *   - NORMAL: Normal operation with full lane tracking
 *   - RECOVERY: Error recovery mode with simplified lane estimation
 *
 * Coordinate System:
 *   - Vehicle Frame: X-forward, Y-left, origin at vehicle center
 *   - Global Frame: World coordinates used for persistent memory storage
 *
 * Memory Architecture:
 *   - lane*_points_global_: Persistent memory in world coordinates
 *   - lane*_points_: Per-frame working memory in vehicle coordinates
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
 */
LaneProcessor::LaneProcessor()
    : current_driveway_(2),       // 기본 driveway: 중앙 차선 (lane2-lane3 사이)
      prev_driveway_(2),
      total_processed_frames_(0),
      lane1_is_generated_(false), // 실측 기반 여부 (false = 실측, true = 외분으로 생성)
      lane2_is_generated_(false),
      lane3_is_generated_(false),
      lane4_is_generated_(false),
      lane1_is_primary_(false),   // Primary lane 여부
      lane2_is_primary_(false),
      lane3_is_primary_(false),
      lane4_is_primary_(false),
      primary_count_(0),
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
      driveway_change_candidate_count_(0),
      last_reset_reason_(LaneErrorType::NONE),
      last_reset_detail_("")
{
    // Primary lane IDs 초기화
    for (int i = 0; i < 4; ++i) {
        primary_lane_ids_[i] = 0;
    }
    // Lane ID 설정
    lane1_polyfit_.id = "lane1";
    lane2_polyfit_.id = "lane2";
    lane3_polyfit_.id = "lane3";
    lane4_polyfit_.id = "lane4";

    // Lane point 메모리 벡터 미리 할당 (성능 최적화)
    static constexpr size_t INITIAL_POINT_BUFFER_CAPACITY = 1000;
    lane1_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane2_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane3_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane4_points_.reserve(INITIAL_POINT_BUFFER_CAPACITY);

    // Global memory 벡터 미리 할당
    lane1_points_global_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane2_points_global_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane3_points_global_.reserve(INITIAL_POINT_BUFFER_CAPACITY);
    lane4_points_global_.reserve(INITIAL_POINT_BUFFER_CAPACITY);

    // Initialize recovery vehicle state to origin
    prev_vehicle_state_recovery_.x = 0.0;
    prev_vehicle_state_recovery_.y = 0.0;
    prev_vehicle_state_recovery_.yaw = 0.0;
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
 *   - INIT/NORMAL: Use global/vehicle frame memory architecture
 *   - RECOVERY: Simplified left/right lane reconstruction
 */
interface::PolyfitLanes LaneProcessor::Process(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state) {

    // -------------------------------------------------------------------------
    // 상태에 따라 분기 처리
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
 * @details New pipeline:
 *   1. BuildVehicleMemoryFromGlobal - global -> vehicle 변환
 *   2. 1차 폴리핏 (클러스터링 기준)
 *   3. 새 포인트를 폴리핏 기반으로 메모리에 누적
 *   4. 최종 폴리핏 생성
 *   5. 균등 샘플 생성 -> 메모리에 추가
 *   6. 다운샘플링
 *   7. UpdateGlobalMemoryFromVehicle - vehicle -> global 변환
 *   8. 후처리 (driveway, center lane, 에러 체크)
 */
interface::PolyfitLanes LaneProcessor::ProcessNormalMode(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state) {

    std::vector<interface::Point2D> input_points = input_lane_data.point;

    const bool is_initialization_phase =
        (total_processed_frames_ < INITIALIZATION_FRAMES);

    // =========================================================================
    // INIT 단계 처리
    // =========================================================================
    if (is_initialization_phase) {
        // ---------------------------------------------------------------------
        // INIT Step 1: 입력 포인트를 Y 기준으로 분류하여 vehicle frame에 저장
        // ---------------------------------------------------------------------
        ClassifyAndStoreLanesByY(input_points, vehicle_state);

        // ---------------------------------------------------------------------
        // INIT Step 2: vehicle frame 포인트를 global frame으로 변환하여 저장
        // ---------------------------------------------------------------------
        for (size_t i = 0; i < lane2_points_.size(); ++i) {
            interface::Point2D global_pt =
                VehicleToGlobalFrame(lane2_points_[i], vehicle_state);
            lane2_points_global_.push_back(global_pt);
        }
        for (size_t i = 0; i < lane3_points_.size(); ++i) {
            interface::Point2D global_pt =
                VehicleToGlobalFrame(lane3_points_[i], vehicle_state);
            lane3_points_global_.push_back(global_pt);
        }

        // ---------------------------------------------------------------------
        // INIT Step 3: global 메모리를 vehicle frame으로 변환하여 폴리핏 수행
        // ---------------------------------------------------------------------
        BuildVehicleMemoryFromGlobal(vehicle_state);

        // 범위 필터링
        FilterStoredPointsByRange(vehicle_state);

        // 다운샘플링
        DownsampleLanePoints();

        // 폴리핏
        FitLanePolynomials(vehicle_state);

        // ---------------------------------------------------------------------
        // INIT Step 4: 프레임 카운터 증가 및 상태 전이 체크
        // ---------------------------------------------------------------------
        total_processed_frames_++;

        const bool should_transition_to_normal =
            (tracking_state_ == LaneTrackingState::INIT) &&
            (total_processed_frames_ >= INITIALIZATION_FRAMES);

        if (should_transition_to_normal) {
            tracking_state_ = LaneTrackingState::NORMAL;
            std::cout << "[LaneProcessor] State transition: INIT -> NORMAL" << std::endl;
        }

        // 결과 반환
        interface::PolyfitLanes result;
        result.frame_id = input_lane_data.frame_id;
        result.polyfitlanes.push_back(lane1_polyfit_);
        result.polyfitlanes.push_back(lane2_polyfit_);
        result.polyfitlanes.push_back(lane3_polyfit_);
        result.polyfitlanes.push_back(lane4_polyfit_);
        result.polyfitlanes.push_back(ego_center_lane_);
        return result;
    }

    // =========================================================================
    // NORMAL 모드 파이프라인
    //
    // 흐름: "샘플만 시작" → "실측 추가" → "폴리핏 + primary + 외분/내분"
    //       → "메모리를 샘플로 리셋" → "global에 저장"
    // =========================================================================

    // -------------------------------------------------------------------------
    // Step 0: 이번 프레임 시작 시 real_points 초기화
    // -------------------------------------------------------------------------
    lane1_real_points_.clear();
    lane2_real_points_.clear();
    lane3_real_points_.clear();
    lane4_real_points_.clear();

    // -------------------------------------------------------------------------
    // Step 1: Global 메모리 -> Vehicle frame 작업 메모리 생성
    //         이 시점의 lane*_points_에는 이전 프레임에서 만든 "균등 샘플"만 존재
    // -------------------------------------------------------------------------
    BuildVehicleMemoryFromGlobal(vehicle_state);

    // 범위 필터링 (vehicle frame 기준)
    FilterStoredPointsByRange(vehicle_state);

    // -------------------------------------------------------------------------
    // Step 2: 새 입력 포인트를 폴리핏 기반으로 클러스터링 + 메모리 누적
    //         ClassifyAndStoreLanesByPolyfit 안에서:
    //         - lane*_points_ 에 push_back
    //         - lane*_real_points_ 에도 push_back
    // -------------------------------------------------------------------------
    ClassifyAndStoreLanesByPolyfit(input_points, vehicle_state);

    // -------------------------------------------------------------------------
    // Step 3: 누적된 메모리(샘플 + 실측) 기반으로 폴리피팅 + primary + 외분/내분
    //         FitLanePolynomials 안에서:
    //         - SavePreviousPolyfits()
    //         - 개별 피팅 (FitSingleLane)
    //         - DeterminePrimaryLanesFromRealPoints()
    //         - RebuildNonPrimaryLanesFromPrimary()
    //         - DetermineCurrentDriveway(), EnforceLaneOrdering(), GenerateEgoCenterLane()
    //         - RefreshLaneMemoryWithUniformSamples()
    //         까지 실행된다.
    // -------------------------------------------------------------------------
    FitLanePolynomials(vehicle_state);

    // -------------------------------------------------------------------------
    // Step 4: RefreshLaneMemoryWithUniformSamples()까지 끝난 후의 lane*_points_는
    //         "최종 폴리핏 균등 샘플"만 남아있는 상태이다.
    //         이제 이것을 global 메모리로 저장한다.
    // -------------------------------------------------------------------------
    UpdateGlobalMemoryFromVehicle(vehicle_state);

    // -------------------------------------------------------------------------
    // Step 8: 에러 체크 (NORMAL 모드에서만)
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
    // Step 9: 프레임 카운터 증가
    // -------------------------------------------------------------------------
    total_processed_frames_++;

    // -------------------------------------------------------------------------
    // Step 10: 결과 반환
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
        CompensateLanePoints(recovery_left_points_, prev_vehicle_state_recovery_, vehicle_state);
        CompensateLanePoints(recovery_right_points_, prev_vehicle_state_recovery_, vehicle_state);

        // 범위 필터링 (MEMORY_X_MIN ~ MEMORY_X_MAX)
        FilterPointsByXRange(recovery_left_points_);
        FilterPointsByXRange(recovery_right_points_);
    }

    // -------------------------------------------------------------------------
    // Step 2: 입력 포인트를 left/right로 분류 (Y 부호 기준)
    // -------------------------------------------------------------------------
    for (size_t i = 0; i < input_points.size(); ++i) {
        const interface::Point2D& input_pt = input_points[i];

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

        // Vehicle frame 포인트를 global로 변환하여 저장
        lane2_points_global_.clear();
        lane3_points_global_.clear();
        for (size_t i = 0; i < lane2_points_.size(); ++i) {
            interface::Point2D global_pt =
                VehicleToGlobalFrame(lane2_points_[i], vehicle_state);
            lane2_points_global_.push_back(global_pt);
        }
        for (size_t i = 0; i < lane3_points_.size(); ++i) {
            interface::Point2D global_pt =
                VehicleToGlobalFrame(lane3_points_[i], vehicle_state);
            lane3_points_global_.push_back(global_pt);
        }

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
    prev_vehicle_state_recovery_ = vehicle_state;

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
// New Pipeline Helper Functions
// ============================================================================

/**
 * @brief Builds vehicle frame working memory from global memory
 *
 * @param current_state Current vehicle pose in global frame
 *
 * @details
 * 각 lane*_points_global_의 포인트를 GlobalToVehicleFrame()으로 변환하여
 * lane*_points_에 저장한다. 이전 내용은 clear된다.
 */
void LaneProcessor::BuildVehicleMemoryFromGlobal(
    const interface::VehicleState& current_state) {

    // Clear vehicle frame working memory
    lane1_points_.clear();
    lane2_points_.clear();
    lane3_points_.clear();
    lane4_points_.clear();

    // Transform lane1 global -> vehicle
    for (size_t i = 0; i < lane1_points_global_.size(); ++i) {
        interface::Point2D vehicle_pt =
            GlobalToVehicleFrame(lane1_points_global_[i], current_state);
        lane1_points_.push_back(vehicle_pt);
    }

    // Transform lane2 global -> vehicle
    for (size_t i = 0; i < lane2_points_global_.size(); ++i) {
        interface::Point2D vehicle_pt =
            GlobalToVehicleFrame(lane2_points_global_[i], current_state);
        lane2_points_.push_back(vehicle_pt);
    }

    // Transform lane3 global -> vehicle
    for (size_t i = 0; i < lane3_points_global_.size(); ++i) {
        interface::Point2D vehicle_pt =
            GlobalToVehicleFrame(lane3_points_global_[i], current_state);
        lane3_points_.push_back(vehicle_pt);
    }

    // Transform lane4 global -> vehicle
    for (size_t i = 0; i < lane4_points_global_.size(); ++i) {
        interface::Point2D vehicle_pt =
            GlobalToVehicleFrame(lane4_points_global_[i], current_state);
        lane4_points_.push_back(vehicle_pt);
    }
}

/**
 * @brief Updates global memory from vehicle frame working memory
 *
 * @param current_state Current vehicle pose in global frame
 *
 * @details
 * 각 lane*_points_의 포인트를 VehicleToGlobalFrame()으로 변환하여
 * lane*_points_global_에 저장한다. 이전 내용은 clear된다.
 */
void LaneProcessor::UpdateGlobalMemoryFromVehicle(
    const interface::VehicleState& current_state) {

    // Clear global memory
    lane1_points_global_.clear();
    lane2_points_global_.clear();
    lane3_points_global_.clear();
    lane4_points_global_.clear();

    // Transform lane1 vehicle -> global
    for (size_t i = 0; i < lane1_points_.size(); ++i) {
        interface::Point2D global_pt =
            VehicleToGlobalFrame(lane1_points_[i], current_state);
        lane1_points_global_.push_back(global_pt);
    }

    // Transform lane2 vehicle -> global
    for (size_t i = 0; i < lane2_points_.size(); ++i) {
        interface::Point2D global_pt =
            VehicleToGlobalFrame(lane2_points_[i], current_state);
        lane2_points_global_.push_back(global_pt);
    }

    // Transform lane3 vehicle -> global
    for (size_t i = 0; i < lane3_points_.size(); ++i) {
        interface::Point2D global_pt =
            VehicleToGlobalFrame(lane3_points_[i], current_state);
        lane3_points_global_.push_back(global_pt);
    }

    // Transform lane4 vehicle -> global
    for (size_t i = 0; i < lane4_points_.size(); ++i) {
        interface::Point2D global_pt =
            VehicleToGlobalFrame(lane4_points_[i], current_state);
        lane4_points_global_.push_back(global_pt);
    }
}

/**
 * @brief Generates uniform samples from polyfit and adds to memory
 *
 * @param lane_polyfit Source polynomial for sampling
 * @param lane_points_vehicle Destination buffer (vehicle frame)
 * @param x_min Minimum X for sampling range
 * @param x_max Maximum X for sampling range
 * @param x_step Sampling interval
 *
 * @details
 * x를 [x_min, x_max] 구간에서 x_step 간격으로 샘플링하여
 * y = EvaluatePolynomial()로 계산한 포인트를 메모리에 추가한다.
 */
void LaneProcessor::GenerateUniformSamplesForLane(
    const interface::PolyfitLane& lane_polyfit,
    std::vector<interface::Point2D>& lane_points_vehicle,
    double x_min,
    double x_max,
    double x_step) {

    // polyfit x_start/x_end가 유효한 경우에만 샘플링
    const bool polyfit_has_valid_range =
        (lane_polyfit.x_end > lane_polyfit.x_start);
    if (!polyfit_has_valid_range) {
        return;
    }

    // 샘플링 범위를 polyfit 유효 범위와 교집합으로 제한
    const double sample_x_start = std::max(x_min, lane_polyfit.x_start);
    const double sample_x_end = std::min(x_max, lane_polyfit.x_end);

    if (sample_x_end <= sample_x_start) {
        return;
    }

    // 균등 간격 샘플링
    for (double x = sample_x_start; x <= sample_x_end; x += x_step) {
        const double y = EvaluatePolynomial(lane_polyfit, x);

        if (std::isfinite(y)) {
            interface::Point2D sample_pt;
            sample_pt.x = x;
            sample_pt.y = y;
            lane_points_vehicle.push_back(sample_pt);
        }
    }
}

/**
 * @brief Transforms lane points from one vehicle frame to another
 *
 * @param[in,out] lane_points Points to transform (modified in place)
 * @param[in] from_state Previous vehicle pose (source frame)
 * @param[in] to_state Current vehicle pose (target frame)
 *
 * @details Used for RECOVERY mode ego-motion compensation
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

    for (size_t i = 0; i < lane_points.size(); ++i) {
        interface::Point2D& point = lane_points[i];

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
 */
void LaneProcessor::ClassifyAndStoreLanesByY(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // Unused parameter

    for (size_t i = 0; i < input_points.size(); ++i) {
        const interface::Point2D& vehicle_point = input_points[i];

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
 * 3프레임 이후에는 기존 polyfit을 기준으로 분류
 */
void LaneProcessor::ClassifyAndStoreLanesByPolyfit(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;  // Unused parameter

    for (size_t i = 0; i < input_points.size(); ++i) {
        const interface::Point2D& vehicle_point = input_points[i];
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
        // Step 3: 가장 가까운 driveway 선택
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
        // ---------------------------------------------------------------------
        const bool is_point_left_of_center = (point_y > best_candidate->center_y);
        const int assigned_lane_index = is_point_left_of_center
            ? best_candidate->left_lane_index
            : best_candidate->right_lane_index;

        // ---------------------------------------------------------------------
        // Step 5: 해당 레인 버퍼에 포인트 추가
        //         - lane*_points_: 폴리피팅용 메모리 (샘플 + 실측)
        //         - lane*_real_points_: primary 판단용 (이번 프레임 실측만)
        // ---------------------------------------------------------------------
        switch (assigned_lane_index) {
            case 1:
                lane1_points_.push_back(vehicle_point);
                lane1_real_points_.push_back(vehicle_point);
                break;
            case 2:
                lane2_points_.push_back(vehicle_point);
                lane2_real_points_.push_back(vehicle_point);
                break;
            case 3:
                lane3_points_.push_back(vehicle_point);
                lane3_real_points_.push_back(vehicle_point);
                break;
            case 4:
                lane4_points_.push_back(vehicle_point);
                lane4_real_points_.push_back(vehicle_point);
                break;
            default:
                break;
        }
    }
}

// ============================================================================
// Extrapolation Utilities (외분을 통한 레인 생성)
// ============================================================================

/**
 * @brief Extrapolates a lane using 1:1 ratio (한 칸 외분)
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

void LaneProcessor::RebuildOuterLanesOnDrivewayChange() {
    // Driveway 변경이 없으면 아무것도 하지 않음
    const bool driveway_unchanged = (prev_driveway_ == current_driveway_);
    if (driveway_unchanged) {
        return;
    }

    std::cout << "[LaneProcessor] Driveway changed: " << prev_driveway_
              << " -> " << current_driveway_ << std::endl;

    if (current_driveway_ == 1) {
        lane3_points_.clear();
        lane4_points_.clear();

        ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
        lane3_is_generated_ = true;

        ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

    } else if (current_driveway_ == 2) {
        lane1_points_.clear();
        lane4_points_.clear();

        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;

        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

    } else if (current_driveway_ == 3) {
        lane1_points_.clear();
        lane2_points_.clear();

        ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
        lane2_is_generated_ = true;

        ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;
    }
}

// ============================================================================
// Update Polyfits From Memory Only
// ============================================================================

void LaneProcessor::UpdatePolyfitsFromMemoryOnly() {
    bool lane1_has_real_points = false;
    bool lane2_has_real_points = false;
    bool lane3_has_real_points = false;
    bool lane4_has_real_points = false;

    FitSingleLane(lane1_points_, lane1_polyfit_, lane1_has_real_points, false);
    FitSingleLane(lane2_points_, lane2_polyfit_, lane2_has_real_points, false);
    FitSingleLane(lane3_points_, lane3_polyfit_, lane3_has_real_points, false);
    FitSingleLane(lane4_points_, lane4_polyfit_, lane4_has_real_points, false);
}

// ============================================================================
// Polynomial Fitting - Main Entry Point
// ============================================================================

void LaneProcessor::FitLanePolynomials(
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;

    // -------------------------------------------------------------------------
    // Step 1: 이전 polyfit 저장 (롤백 가능성 대비)
    // -------------------------------------------------------------------------
    SavePreviousPolyfits();

    // -------------------------------------------------------------------------
    // Step 2: 초기화 프레임 처리 (0~1프레임)
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
            ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
            ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        }

        // Generated 플래그 설정
        lane1_is_generated_ = true;
        lane2_is_generated_ = !lane2_has_points;
        lane3_is_generated_ = !lane3_has_points;
        lane4_is_generated_ = true;

        // Driveway 결정 및 저장
        DetermineCurrentDriveway();
        prev_driveway_ = current_driveway_;

        // Ego center lane 생성
        GenerateEgoCenterLane();
        return;
    }

    // -------------------------------------------------------------------------
    // Step 3: 각 레인 독립적으로 피팅 (현재 lane*_points_ = 샘플 + 실측)
    // -------------------------------------------------------------------------
    bool lane1_has_points = false;
    bool lane2_has_points = false;
    bool lane3_has_points = false;
    bool lane4_has_points = false;

    FitSingleLane(lane1_points_, lane1_polyfit_, lane1_has_points, false);
    FitSingleLane(lane2_points_, lane2_polyfit_, lane2_has_points, false);
    FitSingleLane(lane3_points_, lane3_polyfit_, lane3_has_points, false);
    FitSingleLane(lane4_points_, lane4_polyfit_, lane4_has_points, false);

    // -------------------------------------------------------------------------
    // Step 4: 이번 프레임 "실측" 기준으로 primary 2개 선택 (x_max 큰 순)
    // -------------------------------------------------------------------------
    DeterminePrimaryLanesFromRealPoints();

    // -------------------------------------------------------------------------
    // Step 5: Primary 조합에 따라 non-primary 레인 외분/내분으로 생성
    // -------------------------------------------------------------------------
    RebuildNonPrimaryLanesFromPrimary();

    // -------------------------------------------------------------------------
    // Step 6: Driveway 판단, Lane ordering, Ego center lane 생성
    // -------------------------------------------------------------------------
    DetermineCurrentDriveway();
    EnforceLaneOrdering();
    GenerateEgoCenterLane();

    // -------------------------------------------------------------------------
    // Step 7: 최종 폴리핏 기반으로 메모리를 "균등 샘플만" 남기도록 재구성
    // -------------------------------------------------------------------------
    RefreshLaneMemoryWithUniformSamples();
}

// ============================================================================
// Previous Polyfit Storage (롤백용)
// ============================================================================

void LaneProcessor::SavePreviousPolyfits() {
    prev_lane1_polyfit_ = lane1_polyfit_;
    prev_lane2_polyfit_ = lane2_polyfit_;
    prev_lane3_polyfit_ = lane3_polyfit_;
    prev_lane4_polyfit_ = lane4_polyfit_;
}

// ============================================================================
// Lane Ordering Enforcement
// ============================================================================

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
// Error Detection
// ============================================================================

bool LaneProcessor::HasLaneDetectionError() {
    // HardReset active 상태에서는 추가 HardReset 금지
    if (hard_reset_active_) {
        return false;
    }

    bool soft_error_detected = false;
    bool fatal_error_detected = false;

    // 에러 목록 초기화
    detected_errors_.clear();
    std::string error_detail;

    // -------------------------------------------------------------------------
    // Error Check 1: NaN/Inf 체크 -> fatal 에러
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
        detected_errors_.push_back(LaneErrorType::NAN_INF_COEFFICIENTS);
        std::string nan_lanes;
        if (!lane1_has_finite_coeffs) nan_lanes += "lane1 ";
        if (!lane2_has_finite_coeffs) nan_lanes += "lane2 ";
        if (!lane3_has_finite_coeffs) nan_lanes += "lane3 ";
        if (!lane4_has_finite_coeffs) nan_lanes += "lane4 ";
        error_detail += "[NaN/Inf in " + nan_lanes + "] ";
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
        detected_errors_.push_back(LaneErrorType::LANE_ORDERING_VIOLATION);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        oss << "[Ordering: y1=" << y1_at_origin << " y2=" << y2_at_origin
            << " y3=" << y3_at_origin << " y4=" << y4_at_origin;
        if (!lane1_left_of_lane2) oss << " (y1<=y2)";
        if (!lane2_left_of_lane3) oss << " (y2<=y3)";
        if (!lane3_left_of_lane4) oss << " (y3<=y4)";
        oss << "] ";
        error_detail += oss.str();
    }

    // -------------------------------------------------------------------------
    // Error Check 3: 인접 레인 겹침(overlap) 체크 -> fatal 에러
    // -------------------------------------------------------------------------
    const bool lane1_lane2_overlap = AreLanesTooClose(lane1_polyfit_, lane2_polyfit_);
    const bool lane2_lane3_overlap = AreLanesTooClose(lane2_polyfit_, lane3_polyfit_);
    const bool lane3_lane4_overlap = AreLanesTooClose(lane3_polyfit_, lane4_polyfit_);

    const bool any_lanes_overlap =
        lane1_lane2_overlap || lane2_lane3_overlap || lane3_lane4_overlap;

    if (any_lanes_overlap) {
        fatal_error_detected = true;
        detected_errors_.push_back(LaneErrorType::LANE_OVERLAP);
        std::string overlap_pairs;
        if (lane1_lane2_overlap) overlap_pairs += "lane1-lane2 ";
        if (lane2_lane3_overlap) overlap_pairs += "lane2-lane3 ";
        if (lane3_lane4_overlap) overlap_pairs += "lane3-lane4 ";
        error_detail += "[Overlap: " + overlap_pairs + "] ";
    }

    // -------------------------------------------------------------------------
    // Step 4: 에러 카운터 갱신 (히스테리시스)
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
    // Step 5: HardReset 트리거 결정
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
    // Step 6: HardReset 쿨다운 검사
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

    // 에러 원인 저장
    if (detected_errors_.size() == 1) {
        last_reset_reason_ = detected_errors_[0];
    } else if (detected_errors_.size() > 1) {
        last_reset_reason_ = LaneErrorType::MULTIPLE_ERRORS;
    } else {
        last_reset_reason_ = LaneErrorType::NONE;
    }
    last_reset_detail_ = error_detail;

    return true;
}

// ============================================================================
// HardReset: 전체 메모리 초기화 및 RECOVERY 모드 진입
// ============================================================================

void LaneProcessor::HardReset() {
    std::cout << "\n[LaneProcessor] ============================================" << std::endl;
    std::cout << "[LaneProcessor] === HardReset triggered ===" << std::endl;
    std::cout << "[LaneProcessor] ============================================" << std::endl;

    // 에러 원인 로깅
    std::cout << "[LaneProcessor] Reason: " << GetErrorTypeString(last_reset_reason_) << std::endl;
    if (!last_reset_detail_.empty()) {
        std::cout << "[LaneProcessor] Detail: " << last_reset_detail_ << std::endl;
    }

    // 복수 에러 발생 시 각각의 에러 타입 출력
    if (detected_errors_.size() > 1) {
        std::cout << "[LaneProcessor] Multiple errors detected (" << detected_errors_.size() << "):" << std::endl;
        for (size_t i = 0; i < detected_errors_.size(); ++i) {
            std::cout << "  - " << GetErrorTypeString(detected_errors_[i]) << std::endl;
        }
    }
    std::cout << "[LaneProcessor] ============================================\n" << std::endl;

    // HardReset 락 활성화 (NORMAL 복귀 전까지 재진입 방지)
    hard_reset_active_ = true;

    // -------------------------------------------------------------------------
    // Step 1: Lane 포인트 메모리 초기화 (vehicle frame)
    // -------------------------------------------------------------------------
    lane1_points_.clear();
    lane2_points_.clear();
    lane3_points_.clear();
    lane4_points_.clear();

    prev_lane1_points_.clear();
    prev_lane2_points_.clear();
    prev_lane3_points_.clear();
    prev_lane4_points_.clear();

    // 실측 포인트 메모리 초기화
    lane1_real_points_.clear();
    lane2_real_points_.clear();
    lane3_real_points_.clear();
    lane4_real_points_.clear();

    // -------------------------------------------------------------------------
    // Step 2: Global 메모리 초기화
    // -------------------------------------------------------------------------
    lane1_points_global_.clear();
    lane2_points_global_.clear();
    lane3_points_global_.clear();
    lane4_points_global_.clear();

    // -------------------------------------------------------------------------
    // Step 2-1: Primary 관련 변수 초기화
    // -------------------------------------------------------------------------
    lane1_is_primary_ = false;
    lane2_is_primary_ = false;
    lane3_is_primary_ = false;
    lane4_is_primary_ = false;
    primary_count_ = 0;
    for (int i = 0; i < 4; ++i) {
        primary_lane_ids_[i] = 0;
    }

    // -------------------------------------------------------------------------
    // Step 3: Polyfit 초기화 (기본 생성자로 리셋)
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
    // Step 4: Generated 플래그 초기화
    // -------------------------------------------------------------------------
    lane1_is_generated_ = false;
    lane2_is_generated_ = false;
    lane3_is_generated_ = false;
    lane4_is_generated_ = false;

    // -------------------------------------------------------------------------
    // Step 5: Driveway 초기화 (기본값 2로 리셋)
    // -------------------------------------------------------------------------
    current_driveway_ = 2;
    prev_driveway_ = 2;

    // -------------------------------------------------------------------------
    // Step 6: RECOVERY 관련 멤버 초기화
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
    // Step 7: 에러 카운터 초기화
    // -------------------------------------------------------------------------
    consecutive_soft_error_frames_ = 0;
    consecutive_fatal_error_frames_ = 0;

    // -------------------------------------------------------------------------
    // Step 8: 프레임 카운터 초기화
    // -------------------------------------------------------------------------
    total_processed_frames_ = 0;

    // -------------------------------------------------------------------------
    // Step 9: 상태 전환 (NORMAL -> RECOVERY)
    // -------------------------------------------------------------------------
    tracking_state_ = LaneTrackingState::RECOVERY;
}

// ============================================================================
// Driveway Determination
// ============================================================================

void LaneProcessor::DetermineCurrentDriveway() {
    static constexpr double DRIVEWAY_EVALUATION_X = 5.0;
    static constexpr int DRIVEWAY_SWITCH_STABLE_FRAMES = 5;

    int best_driveway_id = current_driveway_;
    double best_center_distance = std::numeric_limits<double>::max();
    bool has_valid_candidate = false;

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

    if (!has_valid_candidate) {
        return;
    }

    // 히스테리시스 적용
    if (best_driveway_id == current_driveway_) {
        driveway_change_candidate_id_ = current_driveway_;
        driveway_change_candidate_count_ = 0;
        return;
    }

    if (best_driveway_id == driveway_change_candidate_id_) {
        driveway_change_candidate_count_++;
    } else {
        driveway_change_candidate_id_ = best_driveway_id;
        driveway_change_candidate_count_ = 1;
    }

    if (driveway_change_candidate_count_ >= DRIVEWAY_SWITCH_STABLE_FRAMES) {
        prev_driveway_ = current_driveway_;
        current_driveway_ = driveway_change_candidate_id_;
        driveway_change_candidate_count_ = 0;
    }
}

bool LaneProcessor::TryAddDrivewayCandidate(
    int driveway_id,
    const interface::PolyfitLane& left_lane,
    const interface::PolyfitLane& right_lane,
    double evaluation_x,
    int& out_best_id,
    double& out_best_center_distance) const {

    const double y_left = EvaluatePolynomial(left_lane, evaluation_x);
    const double y_right = EvaluatePolynomial(right_lane, evaluation_x);
    const double driveway_width = std::fabs(y_left - y_right);

    const bool width_too_narrow = (driveway_width < MIN_LANE_WIDTH);
    const bool width_too_wide = (driveway_width > MAX_LANE_WIDTH);

    if (width_too_narrow || width_too_wide) {
        return false;
    }

    const double center_y = 0.5 * (y_left + y_right);
    const double distance_to_ego = std::fabs(center_y);

    if (distance_to_ego < out_best_center_distance) {
        out_best_center_distance = distance_to_ego;
        out_best_id = driveway_id;
    }

    return true;
}

// ============================================================================
// Ego Center Lane Generation
// ============================================================================

void LaneProcessor::GenerateEgoCenterLane() {
    ego_center_lane_.id = "ego_center";
    ego_center_lane_.frame_id = "vehicle";

    if (current_driveway_ == 1) {
        ego_center_lane_.a0 = (lane1_polyfit_.a0 + lane2_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane1_polyfit_.a1 + lane2_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane1_polyfit_.a2 + lane2_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane1_polyfit_.a3 + lane2_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane1_polyfit_.x_start, lane2_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane1_polyfit_.x_end, lane2_polyfit_.x_end);

    } else if (current_driveway_ == 2) {
        ego_center_lane_.a0 = (lane2_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane2_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane2_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane2_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start =
            std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
        ego_center_lane_.x_end =
            std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);

    } else if (current_driveway_ == 3) {
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

void LaneProcessor::DownsampleLanePoints() {
    DownsampleSingleLaneWithTrigger(lane1_points_);
    DownsampleSingleLaneWithTrigger(lane2_points_);
    DownsampleSingleLaneWithTrigger(lane3_points_);
    DownsampleSingleLaneWithTrigger(lane4_points_);
}

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

void LaneProcessor::DownsampleSingleLaneWithTrigger(
    std::vector<interface::Point2D>& lane_points) {

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

void LaneProcessor::FilterStoredPointsByRange(
    const interface::VehicleState& vehicle_state) {

    (void)vehicle_state;

    static constexpr double MAX_LATERAL_MEMORY = 6.0;

    FilterLanePointsByXYRange(lane1_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane2_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane3_points_, MAX_LATERAL_MEMORY);
    FilterLanePointsByXYRange(lane4_points_, MAX_LATERAL_MEMORY);
}

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

double LaneProcessor::EvaluatePolynomial(
    const interface::PolyfitLane& polyfit,
    double x) const {

    return polyfit.a0 +
           polyfit.a1 * x +
           polyfit.a2 * x * x +
           polyfit.a3 * x * x * x;
}

bool LaneProcessor::FitCubicPolynomial(
    const std::vector<interface::Point2D>& points,
    interface::PolyfitLane& out_polyfit) const {

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

    // 정규방정식 해결
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

void LaneProcessor::FitSingleLane(
    std::vector<interface::Point2D>& lane_points,
    interface::PolyfitLane& lane_polyfit,
    bool& has_real_points,
    bool relax_condition) {

    static constexpr size_t MIN_RELIABLE_POINTS = 30;
    static constexpr double MIN_X_RANGE = 20.0;

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
    } else if (has_real_points) {
        // 포인트는 있지만 신뢰도 낮음: 그래도 피팅 시도
        FitCubicPolynomial(lane_points, lane_polyfit);
    }
}

// ============================================================================
// Helper Functions - Range and Validation
// ============================================================================

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

    for (size_t i = 0; i < points.size(); ++i) {
        out_x_min = std::min(out_x_min, points[i].x);
        out_x_max = std::max(out_x_max, points[i].x);
    }

    return out_x_max - out_x_min;
}

bool LaneProcessor::CheckFinitePolyfit(const interface::PolyfitLane& polyfit) const {
    const bool a0_finite = std::isfinite(polyfit.a0);
    const bool a1_finite = std::isfinite(polyfit.a1);
    const bool a2_finite = std::isfinite(polyfit.a2);
    const bool a3_finite = std::isfinite(polyfit.a3);

    return a0_finite && a1_finite && a2_finite && a3_finite;
}

bool LaneProcessor::AreLanesTooClose(
    const interface::PolyfitLane& lane_a,
    const interface::PolyfitLane& lane_b) const {

    const double x_start = std::max({lane_a.x_start, lane_b.x_start, 0.0});
    const double x_end = std::min({lane_a.x_end, lane_b.x_end, 15.0});

    const bool no_overlap_range = (x_end <= x_start);
    if (no_overlap_range) {
        return false;
    }

    static constexpr int NUM_SAMPLE_POINTS = 10;
    static constexpr double OVERLAP_THRESHOLD = 0.05;

    for (int i = 0; i <= NUM_SAMPLE_POINTS; ++i) {
        const double ratio = static_cast<double>(i) / NUM_SAMPLE_POINTS;
        const double sample_x = x_start + ratio * (x_end - x_start);

        const double y_a = EvaluatePolynomial(lane_a, sample_x);
        const double y_b = EvaluatePolynomial(lane_b, sample_x);
        const double y_difference = std::fabs(y_a - y_b);

        const bool difference_is_valid = std::isfinite(y_difference);
        if (!difference_is_valid) {
            continue;
        }

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

const std::vector<interface::Point2D>& LaneProcessor::GetLane1Points() const {
    return lane1_points_;
}

const std::vector<interface::Point2D>& LaneProcessor::GetLane2Points() const {
    return lane2_points_;
}

const std::vector<interface::Point2D>& LaneProcessor::GetLane3Points() const {
    return lane3_points_;
}

const std::vector<interface::Point2D>& LaneProcessor::GetLane4Points() const {
    return lane4_points_;
}

const interface::PolyfitLane& LaneProcessor::GetLane1Polyfit() const {
    return lane1_polyfit_;
}

const interface::PolyfitLane& LaneProcessor::GetLane2Polyfit() const {
    return lane2_polyfit_;
}

const interface::PolyfitLane& LaneProcessor::GetLane3Polyfit() const {
    return lane3_polyfit_;
}

const interface::PolyfitLane& LaneProcessor::GetLane4Polyfit() const {
    return lane4_polyfit_;
}

const interface::PolyfitLane& LaneProcessor::GetEgoLane() const {
    return ego_center_lane_;
}

interface::PolyfitLane LaneProcessor::GetDriveway1CenterLine() const {
    interface::PolyfitLane center;
    center.a0 = 0.5 * (lane1_polyfit_.a0 + lane2_polyfit_.a0);
    center.a1 = 0.5 * (lane1_polyfit_.a1 + lane2_polyfit_.a1);
    center.a2 = 0.5 * (lane1_polyfit_.a2 + lane2_polyfit_.a2);
    center.a3 = 0.5 * (lane1_polyfit_.a3 + lane2_polyfit_.a3);
    center.x_start = std::max(lane1_polyfit_.x_start, lane2_polyfit_.x_start);
    center.x_end = std::min(lane1_polyfit_.x_end, lane2_polyfit_.x_end);
    return center;
}

interface::PolyfitLane LaneProcessor::GetDriveway2CenterLine() const {
    interface::PolyfitLane center;
    center.a0 = 0.5 * (lane2_polyfit_.a0 + lane3_polyfit_.a0);
    center.a1 = 0.5 * (lane2_polyfit_.a1 + lane3_polyfit_.a1);
    center.a2 = 0.5 * (lane2_polyfit_.a2 + lane3_polyfit_.a2);
    center.a3 = 0.5 * (lane2_polyfit_.a3 + lane3_polyfit_.a3);
    center.x_start = std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
    center.x_end = std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);
    return center;
}

interface::PolyfitLane LaneProcessor::GetDriveway3CenterLine() const {
    interface::PolyfitLane center;
    center.a0 = 0.5 * (lane3_polyfit_.a0 + lane4_polyfit_.a0);
    center.a1 = 0.5 * (lane3_polyfit_.a1 + lane4_polyfit_.a1);
    center.a2 = 0.5 * (lane3_polyfit_.a2 + lane4_polyfit_.a2);
    center.a3 = 0.5 * (lane3_polyfit_.a3 + lane4_polyfit_.a3);
    center.x_start = std::max(lane3_polyfit_.x_start, lane4_polyfit_.x_start);
    center.x_end = std::min(lane3_polyfit_.x_end, lane4_polyfit_.x_end);
    return center;
}

int LaneProcessor::GetCurrentDriveway() const {
    return current_driveway_;
}

LaneErrorType LaneProcessor::GetLastResetReason() const {
    return last_reset_reason_;
}

std::string LaneProcessor::GetLastResetDetail() const {
    return last_reset_detail_;
}

std::string LaneProcessor::GetErrorTypeString(LaneErrorType error_type) {
    switch (error_type) {
        case LaneErrorType::NONE:
            return "NONE";
        case LaneErrorType::NAN_INF_COEFFICIENTS:
            return "NAN_INF_COEFFICIENTS (Polyfit coefficients contain NaN/Inf)";
        case LaneErrorType::LANE_ORDERING_VIOLATION:
            return "LANE_ORDERING_VIOLATION (y1 > y2 > y3 > y4 order violated)";
        case LaneErrorType::LANE_OVERLAP:
            return "LANE_OVERLAP (Adjacent lanes too close, < 5cm)";
        case LaneErrorType::ABNORMAL_LANE_WIDTH:
            return "ABNORMAL_LANE_WIDTH (Lane width out of range)";
        case LaneErrorType::MULTIPLE_ERRORS:
            return "MULTIPLE_ERRORS (Multiple error types detected)";
        default:
            return "UNKNOWN_ERROR";
    }
}

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

bool LaneProcessor::IsLane1Valid() const {
    return lane1_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

bool LaneProcessor::IsLane2Valid() const {
    return lane2_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

bool LaneProcessor::IsLane3Valid() const {
    return lane3_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

bool LaneProcessor::IsLane4Valid() const {
    return lane4_points_.size() >= RELIABLE_CLUSTER_MIN_POINTS;
}

// ============================================================================
// ROS Message Conversion (visualization용)
// ============================================================================

std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane1PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane1_points_.size());

    for (size_t i = 0; i < lane1_points_.size(); ++i) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = lane1_points_[i].x;
        ros_point.y = lane1_points_[i].y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane2PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane2_points_.size());

    for (size_t i = 0; i < lane2_points_.size(); ++i) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = lane2_points_[i].x;
        ros_point.y = lane2_points_[i].y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane3PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane3_points_.size());

    for (size_t i = 0; i < lane3_points_.size(); ++i) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = lane3_points_[i].x;
        ros_point.y = lane3_points_[i].y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

std::vector<geometry_msgs::msg::Point>
LaneProcessor::getLane4PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {

    (void)vehicle_state;

    std::vector<geometry_msgs::msg::Point> ros_points;
    ros_points.reserve(lane4_points_.size());

    for (size_t i = 0; i < lane4_points_.size(); ++i) {
        geometry_msgs::msg::Point ros_point;
        ros_point.x = lane4_points_[i].x;
        ros_point.y = lane4_points_[i].y;
        ros_point.z = 0.0;
        ros_points.push_back(ros_point);
    }

    return ros_points;
}

// ============================================================================
// Primary Lane Determination (실측 포인트 기반)
// ============================================================================

void LaneProcessor::DeterminePrimaryLanesFromRealPoints() {
    // -------------------------------------------------------------------------
    // Step 1: 모든 primary 플래그 초기화
    // -------------------------------------------------------------------------
    lane1_is_primary_ = false;
    lane2_is_primary_ = false;
    lane3_is_primary_ = false;
    lane4_is_primary_ = false;
    primary_count_ = 0;

    for (int i = 0; i < 4; ++i) {
        primary_lane_ids_[i] = 0;
    }

    // -------------------------------------------------------------------------
    // Step 2: 각 레인의 실측 포인트 개수와 x_max 계산
    // -------------------------------------------------------------------------
    struct LaneCandidate {
        int lane_id;
        size_t point_count;
        double x_max;
    };

    LaneCandidate candidates[4];

    // Lane 1
    candidates[0].lane_id = 1;
    candidates[0].point_count = lane1_real_points_.size();
    candidates[0].x_max = -1e9;
    for (size_t i = 0; i < lane1_real_points_.size(); ++i) {
        if (lane1_real_points_[i].x > candidates[0].x_max) {
            candidates[0].x_max = lane1_real_points_[i].x;
        }
    }

    // Lane 2
    candidates[1].lane_id = 2;
    candidates[1].point_count = lane2_real_points_.size();
    candidates[1].x_max = -1e9;
    for (size_t i = 0; i < lane2_real_points_.size(); ++i) {
        if (lane2_real_points_[i].x > candidates[1].x_max) {
            candidates[1].x_max = lane2_real_points_[i].x;
        }
    }

    // Lane 3
    candidates[2].lane_id = 3;
    candidates[2].point_count = lane3_real_points_.size();
    candidates[2].x_max = -1e9;
    for (size_t i = 0; i < lane3_real_points_.size(); ++i) {
        if (lane3_real_points_[i].x > candidates[2].x_max) {
            candidates[2].x_max = lane3_real_points_[i].x;
        }
    }

    // Lane 4
    candidates[3].lane_id = 4;
    candidates[3].point_count = lane4_real_points_.size();
    candidates[3].x_max = -1e9;
    for (size_t i = 0; i < lane4_real_points_.size(); ++i) {
        if (lane4_real_points_[i].x > candidates[3].x_max) {
            candidates[3].x_max = lane4_real_points_[i].x;
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: 신뢰 가능한 후보만 필터링 (포인트 개수 >= RELIABLE_CLUSTER_MIN_POINTS)
    // -------------------------------------------------------------------------
    LaneCandidate reliable_candidates[4];
    int reliable_count = 0;

    for (int i = 0; i < 4; ++i) {
        if (candidates[i].point_count >= RELIABLE_CLUSTER_MIN_POINTS) {
            reliable_candidates[reliable_count] = candidates[i];
            reliable_count++;
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: x_max 기준 내림차순 정렬 (버블 정렬)
    // -------------------------------------------------------------------------
    for (int i = 0; i < reliable_count - 1; ++i) {
        for (int j = 0; j < reliable_count - 1 - i; ++j) {
            if (reliable_candidates[j].x_max < reliable_candidates[j + 1].x_max) {
                LaneCandidate temp = reliable_candidates[j];
                reliable_candidates[j] = reliable_candidates[j + 1];
                reliable_candidates[j + 1] = temp;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: 상위 2개를 primary로 선택
    // -------------------------------------------------------------------------
    const int max_primary = (reliable_count < 2) ? reliable_count : 2;

    for (int i = 0; i < max_primary; ++i) {
        const int lane_id = reliable_candidates[i].lane_id;
        primary_lane_ids_[i] = lane_id;
        primary_count_++;

        switch (lane_id) {
            case 1: lane1_is_primary_ = true; break;
            case 2: lane2_is_primary_ = true; break;
            case 3: lane3_is_primary_ = true; break;
            case 4: lane4_is_primary_ = true; break;
            default: break;
        }
    }
}

// ============================================================================
// Non-Primary Lane Reconstruction (외분/내분)
// ============================================================================

void LaneProcessor::RebuildNonPrimaryLanesFromPrimary() {
    // -------------------------------------------------------------------------
    // Case 1: primary_count == 0 (신뢰할 레인이 없음)
    //         모든 레인을 이전 프레임 polyfit 그대로 사용
    // -------------------------------------------------------------------------
    if (primary_count_ == 0) {
        lane1_is_generated_ = true;
        lane2_is_generated_ = true;
        lane3_is_generated_ = true;
        lane4_is_generated_ = true;
        return;
    }

    // -------------------------------------------------------------------------
    // Case 2: primary_count == 1 (신뢰 가능한 레인이 하나만 있음)
    //         primary만 실측 기반, 나머지는 이전 프레임 유지
    // -------------------------------------------------------------------------
    if (primary_count_ == 1) {
        lane1_is_generated_ = !lane1_is_primary_;
        lane2_is_generated_ = !lane2_is_primary_;
        lane3_is_generated_ = !lane3_is_primary_;
        lane4_is_generated_ = !lane4_is_primary_;
        return;
    }

    // -------------------------------------------------------------------------
    // Case 3: primary_count == 2 (정상적인 케이스)
    //         두 primary를 기반으로 나머지 레인을 외분/내분
    // -------------------------------------------------------------------------
    int p1 = primary_lane_ids_[0];
    int p2 = primary_lane_ids_[1];

    // p1 <= p2가 되도록 정렬
    if (p1 > p2) {
        const int temp = p1;
        p1 = p2;
        p2 = temp;
    }

    // Primary 레인은 generated = false
    lane1_is_generated_ = !lane1_is_primary_;
    lane2_is_generated_ = !lane2_is_primary_;
    lane3_is_generated_ = !lane3_is_primary_;
    lane4_is_generated_ = !lane4_is_primary_;

    // -------------------------------------------------------------------------
    // (a) (p1, p2) == (1, 2): lane3 = 1:1 외분, lane4 = 2:1 외분
    // -------------------------------------------------------------------------
    if (p1 == 1 && p2 == 2) {
        ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
        lane3_polyfit_.id = "lane3";
        lane3_is_generated_ = true;

        ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
        lane4_polyfit_.id = "lane4";
        lane4_is_generated_ = true;
    }
    // -------------------------------------------------------------------------
    // (b) (p1, p2) == (2, 3): lane1 = 1:1 외분, lane4 = 1:1 외분
    // -------------------------------------------------------------------------
    else if (p1 == 2 && p2 == 3) {
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_polyfit_.id = "lane1";
        lane1_is_generated_ = true;

        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_polyfit_.id = "lane4";
        lane4_is_generated_ = true;
    }
    // -------------------------------------------------------------------------
    // (c) (p1, p2) == (3, 4): lane2 = 1:1 외분, lane1 = 2:1 외분
    // -------------------------------------------------------------------------
    else if (p1 == 3 && p2 == 4) {
        ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
        lane2_polyfit_.id = "lane2";
        lane2_is_generated_ = true;

        ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_polyfit_.id = "lane1";
        lane1_is_generated_ = true;
    }
    // -------------------------------------------------------------------------
    // (d) (p1, p2) == (1, 3): lane2 = 내분(중간), lane4 = (3*L3 - L1)/2
    // -------------------------------------------------------------------------
    else if (p1 == 1 && p2 == 3) {
        // lane2 = (lane1 + lane3) / 2
        lane2_polyfit_.a0 = (lane1_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        lane2_polyfit_.a1 = (lane1_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        lane2_polyfit_.a2 = (lane1_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        lane2_polyfit_.a3 = (lane1_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        lane2_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane3_polyfit_.x_start);
        lane2_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane3_polyfit_.x_end);
        lane2_polyfit_.id = "lane2";
        lane2_is_generated_ = true;

        // lane4 = (3 * lane3 - lane1) / 2
        lane4_polyfit_.a0 = (3.0 * lane3_polyfit_.a0 - lane1_polyfit_.a0) / 2.0;
        lane4_polyfit_.a1 = (3.0 * lane3_polyfit_.a1 - lane1_polyfit_.a1) / 2.0;
        lane4_polyfit_.a2 = (3.0 * lane3_polyfit_.a2 - lane1_polyfit_.a2) / 2.0;
        lane4_polyfit_.a3 = (3.0 * lane3_polyfit_.a3 - lane1_polyfit_.a3) / 2.0;
        lane4_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane3_polyfit_.x_start);
        lane4_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane3_polyfit_.x_end);
        lane4_polyfit_.id = "lane4";
        lane4_is_generated_ = true;
    }
    // -------------------------------------------------------------------------
    // (e) (p1, p2) == (2, 4): lane3 = 내분(중간), lane1 = 1:1 외분
    // -------------------------------------------------------------------------
    else if (p1 == 2 && p2 == 4) {
        // lane3 = (lane2 + lane4) / 2
        lane3_polyfit_.a0 = (lane2_polyfit_.a0 + lane4_polyfit_.a0) / 2.0;
        lane3_polyfit_.a1 = (lane2_polyfit_.a1 + lane4_polyfit_.a1) / 2.0;
        lane3_polyfit_.a2 = (lane2_polyfit_.a2 + lane4_polyfit_.a2) / 2.0;
        lane3_polyfit_.a3 = (lane2_polyfit_.a3 + lane4_polyfit_.a3) / 2.0;
        lane3_polyfit_.x_start = std::max(lane2_polyfit_.x_start, lane4_polyfit_.x_start);
        lane3_polyfit_.x_end = std::min(lane2_polyfit_.x_end, lane4_polyfit_.x_end);
        lane3_polyfit_.id = "lane3";
        lane3_is_generated_ = true;

        // lane1 = 1:1 외분 (lane2 기준, lane4 참조)
        ExtrapolateLane(lane2_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_polyfit_.id = "lane1";
        lane1_is_generated_ = true;
    }
    // -------------------------------------------------------------------------
    // (f) (p1, p2) == (1, 4): lane2 = 1/3 내분, lane3 = 2/3 내분
    // -------------------------------------------------------------------------
    else if (p1 == 1 && p2 == 4) {
        // lane2 = 2/3 * lane1 + 1/3 * lane4
        lane2_polyfit_.a0 = lane1_polyfit_.a0 * (2.0 / 3.0) + lane4_polyfit_.a0 * (1.0 / 3.0);
        lane2_polyfit_.a1 = lane1_polyfit_.a1 * (2.0 / 3.0) + lane4_polyfit_.a1 * (1.0 / 3.0);
        lane2_polyfit_.a2 = lane1_polyfit_.a2 * (2.0 / 3.0) + lane4_polyfit_.a2 * (1.0 / 3.0);
        lane2_polyfit_.a3 = lane1_polyfit_.a3 * (2.0 / 3.0) + lane4_polyfit_.a3 * (1.0 / 3.0);
        lane2_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane4_polyfit_.x_start);
        lane2_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane4_polyfit_.x_end);
        lane2_polyfit_.id = "lane2";
        lane2_is_generated_ = true;

        // lane3 = 1/3 * lane1 + 2/3 * lane4
        lane3_polyfit_.a0 = lane1_polyfit_.a0 * (1.0 / 3.0) + lane4_polyfit_.a0 * (2.0 / 3.0);
        lane3_polyfit_.a1 = lane1_polyfit_.a1 * (1.0 / 3.0) + lane4_polyfit_.a1 * (2.0 / 3.0);
        lane3_polyfit_.a2 = lane1_polyfit_.a2 * (1.0 / 3.0) + lane4_polyfit_.a2 * (2.0 / 3.0);
        lane3_polyfit_.a3 = lane1_polyfit_.a3 * (1.0 / 3.0) + lane4_polyfit_.a3 * (2.0 / 3.0);
        lane3_polyfit_.x_start = std::max(lane1_polyfit_.x_start, lane4_polyfit_.x_start);
        lane3_polyfit_.x_end = std::min(lane1_polyfit_.x_end, lane4_polyfit_.x_end);
        lane3_polyfit_.id = "lane3";
        lane3_is_generated_ = true;
    }
}

// ============================================================================
// Memory Refresh with Uniform Samples
// ============================================================================

void LaneProcessor::RefreshLaneMemoryWithUniformSamples() {
    // -------------------------------------------------------------------------
    // Step 1: 기존 메모리 클리어
    // -------------------------------------------------------------------------
    lane1_points_.clear();
    lane2_points_.clear();
    lane3_points_.clear();
    lane4_points_.clear();

    // -------------------------------------------------------------------------
    // Step 2: 각 레인 polyfit에서 균등 샘플 생성
    //         polyfit.x_start/x_end와 [MEMORY_X_MIN, MEMORY_X_MAX]의 교집합 사용
    // -------------------------------------------------------------------------
    GenerateUniformSamplesForLane(
        lane1_polyfit_, lane1_points_,
        MEMORY_X_MIN, MEMORY_X_MAX, SAMPLE_X_STEP);

    GenerateUniformSamplesForLane(
        lane2_polyfit_, lane2_points_,
        MEMORY_X_MIN, MEMORY_X_MAX, SAMPLE_X_STEP);

    GenerateUniformSamplesForLane(
        lane3_polyfit_, lane3_points_,
        MEMORY_X_MIN, MEMORY_X_MAX, SAMPLE_X_STEP);

    GenerateUniformSamplesForLane(
        lane4_polyfit_, lane4_points_,
        MEMORY_X_MIN, MEMORY_X_MAX, SAMPLE_X_STEP);

    // -------------------------------------------------------------------------
    // Step 3: 메모리 크기 제한을 위해 다운샘플링
    // -------------------------------------------------------------------------
    DownsampleLanePoints();
}

}  // namespace lane_processor
