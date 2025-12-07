#include <algorithm>
#include <cmath>
#include <tuple>
#include <limits>
#include <iostream>
#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include <geometry_msgs/msg/point.hpp>
#include "lane_processor.hpp"

namespace lane_processor {

LaneProcessor::LaneProcessor()
    : current_driveway_(2),
      prev_driveway_(2),
      total_processed_frames_(0),
      last_log_time_(0.0),
      lane1_is_generated_(false),
      lane2_is_generated_(false),
      lane3_is_generated_(false),
      lane4_is_generated_(false),
      tracking_state_(LaneTrackingState::INIT),
      virtual_left_valid_(false),
      virtual_right_valid_(false),
      recovery_frames_(0),
      recovery_stable_count_(0) {
    lane1_polyfit_.id = "lane1";
    lane2_polyfit_.id = "lane2";
    lane3_polyfit_.id = "lane3";
    lane4_polyfit_.id = "lane4";

    // lane point 메모리 벡터 미리 할당
    lane1_points_.reserve(1000);
    lane2_points_.reserve(1000);
    lane3_points_.reserve(1000);
    lane4_points_.reserve(1000);

    // Initialize prev_vehicle_state
    prev_vehicle_state_.x = 0.0;
    prev_vehicle_state_.y = 0.0;
    prev_vehicle_state_.yaw = 0.0;
}

interface::PolyfitLanes LaneProcessor::Process(const interface::Lane& input_lane_data,
                                                const interface::VehicleState& vehicle_state) {

    // 현재 시간 확인 (2초마다 로그 출력)
    auto now = std::chrono::steady_clock::now();
    double current_time = std::chrono::duration<double>(now.time_since_epoch()).count();
    bool should_log = (current_time - last_log_time_) >= 2.0;

    if (should_log) {
        last_log_time_ = current_time;
        std::string state_str;
        switch (tracking_state_) {
            case LaneTrackingState::INIT: state_str = "INIT"; break;
            case LaneTrackingState::NORMAL: state_str = "NORMAL"; break;
            case LaneTrackingState::RECOVERY: state_str = "RECOVERY"; break;
        }
        std::cout << "\n[LaneProcessor] ===== Frame " << total_processed_frames_
                  << " - State: " << state_str
                  << " - Input points: " << input_lane_data.point.size() << " =====" << std::endl;
    }

    // Step 1: Ego-motion compensation (2프레임 이후부터)
    // 기존 메모리는 유지하고, 좌표만 현재 vehicle frame으로 변환
    if (total_processed_frames_ >= 1) {
        ApplyEgoMotionCompensation(vehicle_state);
        // 범위 밖으로 나간 포인트만 제거 (메모리 전체 재분류 X)
        FilterStoredPointsByRange(vehicle_state);
    }

    // 상태에 따라 분기 처리
    if (tracking_state_ == LaneTrackingState::RECOVERY) {
        return ProcessRecoveryMode(input_lane_data, vehicle_state, should_log);
    } else {
        // INIT / NORMAL은 기존 로직 사용
        return ProcessNormalMode(input_lane_data, vehicle_state, should_log);
    }
}

// ========== NORMAL 모드 처리 (기존 로직) ==========
interface::PolyfitLanes LaneProcessor::ProcessNormalMode(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state,
    bool should_log) {

    std::vector<interface::Point2D> input_points = input_lane_data.point;

    // Step 2: 신규 입력 포인트만 분류하여 기존 메모리에 추가
    if (total_processed_frames_ < INITIALIZATION_FRAMES) {
        ClassifyAndStoreLanesByY(input_points, vehicle_state);
    } else {
        ClassifyAndStoreLanesByPolyfit(input_points, vehicle_state);
    }

    // Step 3: 다운샘플링 (포인트 수 제한)
    DownsampleLanePoints();

    if (should_log) {
        std::cout << "[LaneProcessor] Points memory: "
                  << "L1=" << lane1_points_.size() << " "
                  << "L2=" << lane2_points_.size() << " "
                  << "L3=" << lane3_points_.size() << " "
                  << "L4=" << lane4_points_.size() << std::endl;
    }

    // Polyfit 생성 및 driveway/center lane 갱신
    FitLanePolynomials(vehicle_state, should_log);

    // NORMAL 모드에서만 에러 체크 수행 (INIT 상태에서는 아직 polyfit이 불안정)
    if (tracking_state_ == LaneTrackingState::NORMAL) {
        bool error = HasLaneDetectionError(should_log);
        if (error) {
            if (should_log) {
                std::cout << "[LaneProcessor] Detection error -> HardReset & enter RECOVERY mode" << std::endl;
            }
            HardReset();
            // HardReset 후 빈 결과 반환
            interface::PolyfitLanes result;
            result.frame_id = input_lane_data.frame_id;
            result.polyfitlanes.push_back(lane1_polyfit_);
            result.polyfitlanes.push_back(lane2_polyfit_);
            result.polyfitlanes.push_back(lane3_polyfit_);
            result.polyfitlanes.push_back(lane4_polyfit_);
            result.polyfitlanes.push_back(ego_center_lane_);
            return result;
        }
    }

    if (total_processed_frames_ >= INITIALIZATION_FRAMES && should_log) {
        std::cout << "[LaneProcessor] Current Driveway: " << current_driveway_ << std::endl;
    }

    total_processed_frames_++;

    // INIT -> NORMAL 상태 전환
    if (tracking_state_ == LaneTrackingState::INIT &&
        total_processed_frames_ >= INITIALIZATION_FRAMES) {
        tracking_state_ = LaneTrackingState::NORMAL;
        if (should_log) {
            std::cout << "[LaneProcessor] State transition: INIT -> NORMAL" << std::endl;
        }
    }

    // 다음 프레임을 위해 현재 vehicle state 저장
    prev_vehicle_state_ = vehicle_state;

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

// ========== RECOVERY 모드 처리 ==========
interface::PolyfitLanes LaneProcessor::ProcessRecoveryMode(
    const interface::Lane& input_lane_data,
    const interface::VehicleState& vehicle_state,
    bool should_log) {

    std::vector<interface::Point2D> input_points = input_lane_data.point;

    // RECOVERY 모드에서 ego-motion compensation 적용 (recovery 포인트 버퍼에)
    if (recovery_frames_ >= 1) {
        auto compensate = [this, &vehicle_state](std::vector<interface::Point2D>& pts) {
            for (auto& pt : pts) {
                double cos_prev_yaw = std::cos(prev_vehicle_state_.yaw);
                double sin_prev_yaw = std::sin(prev_vehicle_state_.yaw);

                double global_x = prev_vehicle_state_.x + cos_prev_yaw * pt.x - sin_prev_yaw * pt.y;
                double global_y = prev_vehicle_state_.y + sin_prev_yaw * pt.x + cos_prev_yaw * pt.y;

                double dx = global_x - vehicle_state.x;
                double dy = global_y - vehicle_state.y;

                double cos_curr_yaw = std::cos(-vehicle_state.yaw);
                double sin_curr_yaw = std::sin(-vehicle_state.yaw);

                pt.x = cos_curr_yaw * dx - sin_curr_yaw * dy;
                pt.y = sin_curr_yaw * dx + cos_curr_yaw * dy;
            }
        };
        compensate(recovery_left_points_);
        compensate(recovery_right_points_);

        // 범위 필터링
        auto filter = [this](std::vector<interface::Point2D>& pts) {
            pts.erase(
                std::remove_if(pts.begin(), pts.end(),
                    [this](const interface::Point2D& pt) {
                        return pt.x < MEMORY_X_MIN || pt.x > MEMORY_X_MAX;
                    }),
                pts.end()
            );
        };
        filter(recovery_left_points_);
        filter(recovery_right_points_);
    }

    // (1) 입력 포인트를 left/right로 분류 (y >= 0 -> left, y < 0 -> right)
    for (const auto& pt : input_points) {
        if (pt.x < MEMORY_X_MIN || pt.x > MEMORY_X_MAX) {
            continue;
        }

        if (pt.y >= 0.0) {
            recovery_left_points_.push_back(pt);
        } else {
            recovery_right_points_.push_back(pt);
        }
    }

    // 다운샘플링
    auto downsample = [this](std::vector<interface::Point2D>& pts) {
        if (pts.size() <= MAX_POINTS_PER_LANE) return;

        std::sort(pts.begin(), pts.end(),
                  [](const interface::Point2D& a, const interface::Point2D& b) {
                      return a.x < b.x;
                  });

        std::vector<interface::Point2D> downsampled;
        downsampled.reserve(MAX_POINTS_PER_LANE);

        size_t total = pts.size();
        for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
            size_t idx = (i * total) / MAX_POINTS_PER_LANE;
            if (idx < total) {
                downsampled.push_back(pts[idx]);
            }
        }
        pts = std::move(downsampled);
    };

    downsample(recovery_left_points_);
    downsample(recovery_right_points_);

    // (2) 각 쪽의 포인트가 충분하면 virtual lane 생성
    auto compute_x_range = [](const std::vector<interface::Point2D>& pts) -> double {
        if (pts.empty()) return 0.0;
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        for (const auto& p : pts) {
            x_min = std::min(x_min, p.x);
            x_max = std::max(x_max, p.x);
        }
        return x_max - x_min;
    };

    double left_x_range = compute_x_range(recovery_left_points_);
    double right_x_range = compute_x_range(recovery_right_points_);

    bool left_ok = (recovery_left_points_.size() >= RECOVERY_MIN_POINTS &&
                    left_x_range >= RECOVERY_MIN_X_RANGE);
    bool right_ok = (recovery_right_points_.size() >= RECOVERY_MIN_POINTS &&
                     right_x_range >= RECOVERY_MIN_X_RANGE);

    if (left_ok) {
        if (FitCubicPolynomial(recovery_left_points_, virtual_left_lane_)) {
            virtual_left_valid_ = true;
        }
    }

    if (right_ok) {
        if (FitCubicPolynomial(recovery_right_points_, virtual_right_lane_)) {
            virtual_right_valid_ = true;
        }
    }

    if (should_log) {
        std::cout << "[LaneProcessor] RECOVERY - left_pts=" << recovery_left_points_.size()
                  << " right_pts=" << recovery_right_points_.size()
                  << " left_valid=" << (virtual_left_valid_ ? "Y" : "N")
                  << " right_valid=" << (virtual_right_valid_ ? "Y" : "N")
                  << " recovery_frames=" << recovery_frames_
                  << " stable_count=" << recovery_stable_count_ << std::endl;
    }

    // (3) 현재 프레임의 출력 구성
    if (virtual_left_valid_) {
        lane2_polyfit_ = virtual_left_lane_;
        lane2_polyfit_.id = "lane2";
    }
    if (virtual_right_valid_) {
        lane3_polyfit_ = virtual_right_lane_;
        lane3_polyfit_.id = "lane3";
    }

    // lane1, lane4는 외분으로 생성
    if (virtual_left_valid_ && virtual_right_valid_) {
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_polyfit_.id = "lane1";
        lane1_is_generated_ = true;

        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_polyfit_.id = "lane4";
        lane4_is_generated_ = true;

        // ego center lane 생성
        ego_center_lane_.id = "ego_center";
        ego_center_lane_.frame_id = "vehicle";
        ego_center_lane_.a0 = (lane2_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane2_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane2_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane2_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start = std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
        ego_center_lane_.x_end = std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);
    }

    recovery_frames_++;

    // (4) 안정성 체크: 양쪽 레인이 유효하면 stable count 증가
    if (virtual_left_valid_ && virtual_right_valid_) {
        recovery_stable_count_++;
    } else {
        recovery_stable_count_ = 0;
    }

    // (5) NORMAL 모드로 전환 조건: 충분히 안정화됨
    if (recovery_stable_count_ >= RECOVERY_STABLE_FRAMES) {
        // lane2, lane3 포인트 버퍼에 recovery 포인트 저장
        lane2_points_ = recovery_left_points_;
        lane3_points_ = recovery_right_points_;

        lane2_is_generated_ = false;
        lane3_is_generated_ = false;

        // driveway는 기본 2로 설정
        current_driveway_ = 2;
        prev_driveway_ = 2;

        tracking_state_ = LaneTrackingState::NORMAL;

        if (should_log) {
            std::cout << "[LaneProcessor] State transition: RECOVERY -> NORMAL "
                      << "(stable for " << RECOVERY_STABLE_FRAMES << " frames)" << std::endl;
        }

        // RECOVERY 버퍼 초기화
        recovery_left_points_.clear();
        recovery_right_points_.clear();
        virtual_left_valid_ = false;
        virtual_right_valid_ = false;
        recovery_frames_ = 0;
        recovery_stable_count_ = 0;
    }

    total_processed_frames_++;

    // 다음 프레임을 위해 현재 vehicle state 저장
    prev_vehicle_state_ = vehicle_state;

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

// ========== Ego-motion Compensation ==========
void LaneProcessor::ApplyEgoMotionCompensation(const interface::VehicleState& current_state) {
    auto compensate = [this, &current_state](std::vector<interface::Point2D>& lane_points) {
        for (auto& pt : lane_points) {
            double cos_prev_yaw = std::cos(prev_vehicle_state_.yaw);
            double sin_prev_yaw = std::sin(prev_vehicle_state_.yaw);

            double global_x = prev_vehicle_state_.x + cos_prev_yaw * pt.x - sin_prev_yaw * pt.y;
            double global_y = prev_vehicle_state_.y + sin_prev_yaw * pt.x + cos_prev_yaw * pt.y;

            double dx = global_x - current_state.x;
            double dy = global_y - current_state.y;

            double cos_curr_yaw = std::cos(-current_state.yaw);
            double sin_curr_yaw = std::sin(-current_state.yaw);

            pt.x = cos_curr_yaw * dx - sin_curr_yaw * dy;
            pt.y = sin_curr_yaw * dx + cos_curr_yaw * dy;
        }
    };

    compensate(lane1_points_);
    compensate(lane2_points_);
    compensate(lane3_points_);
    compensate(lane4_points_);
}

// ========== 차량좌표계 y값으로 단순 분류 (초기 2프레임) ==========
void LaneProcessor::ClassifyAndStoreLanesByY(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {
    (void)vehicle_state;

    size_t lane2_count = 0;
    size_t lane3_count = 0;

    for (const auto& vehicle_point : input_points) {
        if (vehicle_point.x < MEMORY_X_MIN || vehicle_point.x > MEMORY_X_MAX) {
            continue;
        }

        if (vehicle_point.y >= 0.0) {
            lane2_points_.push_back(vehicle_point);
            lane2_count++;
        } else {
            lane3_points_.push_back(vehicle_point);
            lane3_count++;
        }
    }

    auto now = std::chrono::steady_clock::now();
    double current_time = std::chrono::duration<double>(now.time_since_epoch()).count();
    bool should_log = (current_time - last_log_time_) >= 2.0;

    if (should_log) {
        std::cout << "[LaneProcessor] ClassifyByY - New points added: "
                  << "L2=" << lane2_count << " "
                  << "L3=" << lane3_count << std::endl;
    }
}

// ========== Driveway 중앙선 기준 분류 (3프레임 이후) ==========
void LaneProcessor::ClassifyAndStoreLanesByPolyfit(
    const std::vector<interface::Point2D>& input_points,
    const interface::VehicleState& vehicle_state) {
    (void)vehicle_state;

    for (const auto& vehicle_point : input_points) {
        double x = vehicle_point.x;
        double y = vehicle_point.y;

        // X 범위 필터링
        if (x < MEMORY_X_MIN || x > MEMORY_X_MAX) {
            continue;
        }

        // 각 lane polyfit에서 해당 x의 y값 계산
        double y1 = EvaluatePolynomial(lane1_polyfit_, x);
        double y2 = EvaluatePolynomial(lane2_polyfit_, x);
        double y3 = EvaluatePolynomial(lane3_polyfit_, x);
        double y4 = EvaluatePolynomial(lane4_polyfit_, x);

        struct DrivewayCandidate {
            int left_lane;
            int right_lane;
            double center_y;
            double dist;
        };

        DrivewayCandidate candidates[3];

        // Driveway 1: lane1 (left) ~ lane2 (right)
        candidates[0].left_lane = 1;
        candidates[0].right_lane = 2;
        candidates[0].center_y = 0.5 * (y1 + y2);
        candidates[0].dist = std::fabs(y - candidates[0].center_y);

        // Driveway 2: lane2 (left) ~ lane3 (right)
        candidates[1].left_lane = 2;
        candidates[1].right_lane = 3;
        candidates[1].center_y = 0.5 * (y2 + y3);
        candidates[1].dist = std::fabs(y - candidates[1].center_y);

        // Driveway 3: lane3 (left) ~ lane4 (right)
        candidates[2].left_lane = 3;
        candidates[2].right_lane = 4;
        candidates[2].center_y = 0.5 * (y3 + y4);
        candidates[2].dist = std::fabs(y - candidates[2].center_y);

        // 가장 가까운 driveway 선택 (threshold 없이)
        const DrivewayCandidate* best = &candidates[0];
        for (int i = 1; i < 3; ++i) {
            if (candidates[i].dist < best->dist) {
                best = &candidates[i];
            }
        }

        // 선택된 driveway의 중앙선 기준으로 부호 비교 후 lane index 결정
        bool is_left = (y > best->center_y);
        int lane_index = is_left ? best->left_lane : best->right_lane;

        switch (lane_index) {
            case 1: lane1_points_.push_back(vehicle_point); break;
            case 2: lane2_points_.push_back(vehicle_point); break;
            case 3: lane3_points_.push_back(vehicle_point); break;
            case 4: lane4_points_.push_back(vehicle_point); break;
            default: break;
        }
    }
}

// ========== 외분 유틸리티 함수 ==========
void LaneProcessor::ExtrapolateLane(const interface::PolyfitLane& base,
                                     const interface::PolyfitLane& ref,
                                     interface::PolyfitLane& target) {
    // 1:1 외분: target = 2*base - ref
    target.a0 = 2.0 * base.a0 - ref.a0;
    target.a1 = 2.0 * base.a1 - ref.a1;
    target.a2 = 2.0 * base.a2 - ref.a2;
    target.a3 = 2.0 * base.a3 - ref.a3;

    double start = std::max(base.x_start, ref.x_start);
    double end = std::min(base.x_end, ref.x_end);
    if (end <= start) {
        start = std::min(base.x_start, ref.x_start);
        end = std::max(base.x_end, ref.x_end);
    }
    target.x_start = start;
    target.x_end = end;
}

void LaneProcessor::ExtrapolateLane2(const interface::PolyfitLane& base,
                                      const interface::PolyfitLane& ref,
                                      interface::PolyfitLane& target) {
    // 2:1 외분: target = 3*base - 2*ref
    target.a0 = 3.0 * base.a0 - 2.0 * ref.a0;
    target.a1 = 3.0 * base.a1 - 2.0 * ref.a1;
    target.a2 = 3.0 * base.a2 - 2.0 * ref.a2;
    target.a3 = 3.0 * base.a3 - 2.0 * ref.a3;

    double start = std::max(base.x_start, ref.x_start);
    double end = std::min(base.x_end, ref.x_end);
    if (end <= start) {
        start = std::min(base.x_start, ref.x_start);
        end = std::max(base.x_end, ref.x_end);
    }
    target.x_start = start;
    target.x_end = end;
}

// ========== Driveway 변경 시 outer lane 재구성 ==========
void LaneProcessor::RebuildOuterLanesOnDrivewayChange(bool should_log) {
    if (prev_driveway_ == current_driveway_) {
        return;  // driveway 변경 없음
    }

    if (should_log) {
        std::cout << "[LaneProcessor] Driveway changed: prev=" << prev_driveway_
                  << ", curr=" << current_driveway_
                  << ", rebuilding outer lanes..." << std::endl;
    }

    if (current_driveway_ == 1) {
        // ego 주변: lane1, lane2 유지
        // outer: lane3, lane4 재구성
        lane3_points_.clear();
        lane4_points_.clear();

        // lane3 = lane2 기준 오른쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
        lane3_is_generated_ = true;

        // lane4 = lane2 기준 오른쪽 두 칸 (2:1 외분)
        ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

        if (should_log) {
            std::cout << "[LaneProcessor] Rebuild L3,L4 from L1,L2 (driveway=1)" << std::endl;
        }

    } else if (current_driveway_ == 2) {
        // ego 주변: lane2, lane3 유지
        // outer: lane1, lane4 재구성
        lane1_points_.clear();
        lane4_points_.clear();

        // lane1 = lane2 기준 왼쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;

        // lane4 = lane3 기준 오른쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;

        if (should_log) {
            std::cout << "[LaneProcessor] Rebuild L1,L4 from L2,L3 (driveway=2)" << std::endl;
        }

    } else if (current_driveway_ == 3) {
        // ego 주변: lane3, lane4 유지
        // outer: lane1, lane2 재구성
        lane1_points_.clear();
        lane2_points_.clear();

        // lane2 = lane3 기준 왼쪽 한 칸 (1:1 외분)
        ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
        lane2_is_generated_ = true;

        // lane1 = lane3 기준 왼쪽 두 칸 (2:1 외분)
        ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;

        if (should_log) {
            std::cout << "[LaneProcessor] Rebuild L1,L2 from L3,L4 (driveway=3)" << std::endl;
        }
    }
}

// ========== Polyfit 생성 (차량좌표계) ==========
void LaneProcessor::FitLanePolynomials(const interface::VehicleState& vehicle_state, bool should_log) {
    (void)vehicle_state;

    SavePreviousPolyfits();

    const size_t MIN_RELIABLE_POINTS = 30;
    const double MIN_X_RANGE = 20.0;

    auto compute_stats = [](const std::vector<interface::Point2D>& pts) {
        double x_min = std::numeric_limits<double>::max();
        double x_max = std::numeric_limits<double>::lowest();
        for (const auto& p : pts) {
            x_min = std::min(x_min, p.x);
            x_max = std::max(x_max, p.x);
        }
        double x_range = pts.empty() ? 0.0 : (x_max - x_min);
        return std::tuple<double, double, double>(x_min, x_max, x_range);
    };

    auto refit_with_samples = [&](const std::vector<interface::Point2D>& pts,
                                  interface::PolyfitLane& poly) {
        if (pts.empty()) return false;

        std::vector<interface::Point2D> refit_pts = pts;
        double start = poly.x_start;
        double end = poly.x_end;
        if (end <= start) {
            start = MEMORY_X_MIN;
            end = MEMORY_X_MAX;
        }
        for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
            double ratio = (MAX_POINTS_PER_LANE == 1)
                               ? 0.0
                               : static_cast<double>(i) / static_cast<double>(MAX_POINTS_PER_LANE - 1);
            double x = start + ratio * (end - start);
            double y = EvaluatePolynomial(poly, x);
            interface::Point2D p;
            p.x = x;
            p.y = y;
            refit_pts.push_back(p);
        }
        return FitCubicPolynomial(refit_pts, poly);
    };

    auto fit_lane = [&](std::vector<interface::Point2D>& pts,
                        interface::PolyfitLane& poly,
                        const std::string& name,
                        bool& has_real_points) {
        has_real_points = !pts.empty();

        double x_min, x_max, x_range;
        std::tie(x_min, x_max, x_range) = compute_stats(pts);
        bool reliable = has_real_points &&
                        pts.size() >= MIN_RELIABLE_POINTS &&
                        x_range >= MIN_X_RANGE;

        bool fitted = false;
        if (reliable) {
            fitted = FitCubicPolynomial(pts, poly);
        } else if (has_real_points) {
            fitted = refit_with_samples(pts, poly);
        }

        if (should_log) {
            std::cout << "[LaneProcessor] " << name
                      << " points=" << pts.size()
                      << " x_range=" << x_range
                      << " reliable=" << (reliable ? "Y" : "N")
                      << " fit=" << (fitted ? "OK" : "SKIP") << std::endl;
        }
    };

    // 초기 2프레임
    if (total_processed_frames_ < INITIALIZATION_FRAMES) {
        bool lane2_has_points = false;
        bool lane3_has_points = false;

        fit_lane(lane2_points_, lane2_polyfit_, "Lane2", lane2_has_points);
        fit_lane(lane3_points_, lane3_polyfit_, "Lane3", lane3_has_points);

        if (lane2_has_points && lane3_has_points) {
            ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
            ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        }

        lane1_is_generated_ = true;
        lane2_is_generated_ = !lane2_has_points;
        lane3_is_generated_ = !lane3_has_points;
        lane4_is_generated_ = true;

        DetermineCurrentDriveway();
        prev_driveway_ = current_driveway_;
        GenerateEgoCenterLane(should_log);
        return;
    }

    // 3프레임 이후
    bool lane1_has_points = !lane1_points_.empty();
    bool lane2_has_points = !lane2_points_.empty();
    bool lane3_has_points = !lane3_points_.empty();
    bool lane4_has_points = !lane4_points_.empty();

    fit_lane(lane1_points_, lane1_polyfit_, "Lane1", lane1_has_points);
    fit_lane(lane2_points_, lane2_polyfit_, "Lane2", lane2_has_points);
    fit_lane(lane3_points_, lane3_polyfit_, "Lane3", lane3_has_points);
    fit_lane(lane4_points_, lane4_polyfit_, "Lane4", lane4_has_points);

    lane1_is_generated_ = !lane1_has_points;
    lane2_is_generated_ = !lane2_has_points;
    lane3_is_generated_ = !lane3_has_points;
    lane4_is_generated_ = !lane4_has_points;

    // ========== 케이스별 외분 생성 ==========

    // Case 1: lane2, lane3만 실측
    if (!lane1_has_points && lane2_has_points && lane3_has_points) {
        ExtrapolateLane(lane2_polyfit_, lane3_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;
        if (should_log) std::cout << "[LaneProcessor] Lane1 generated by extrapolation (L2,L3)" << std::endl;
    }
    if (!lane4_has_points && lane3_has_points && lane2_has_points) {
        ExtrapolateLane(lane3_polyfit_, lane2_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;
        if (should_log) std::cout << "[LaneProcessor] Lane4 generated by extrapolation (L2,L3)" << std::endl;
    }

    // Case 2: lane1, lane2만 실측
    if (lane1_has_points && lane2_has_points && !lane3_has_points && !lane4_has_points) {
        ExtrapolateLane(lane2_polyfit_, lane1_polyfit_, lane3_polyfit_);
        lane3_is_generated_ = true;
        ExtrapolateLane2(lane2_polyfit_, lane1_polyfit_, lane4_polyfit_);
        lane4_is_generated_ = true;
        if (should_log) std::cout << "[LaneProcessor] Lane3,4 generated by extrapolation (L1,L2 only)" << std::endl;
    }

    // Case 3: lane3, lane4만 실측
    if (!lane1_has_points && !lane2_has_points && lane3_has_points && lane4_has_points) {
        ExtrapolateLane(lane3_polyfit_, lane4_polyfit_, lane2_polyfit_);
        lane2_is_generated_ = true;
        ExtrapolateLane2(lane3_polyfit_, lane4_polyfit_, lane1_polyfit_);
        lane1_is_generated_ = true;
        if (should_log) std::cout << "[LaneProcessor] Lane1,2 generated by extrapolation (L3,L4 only)" << std::endl;
    }

    // Case 4: 단일 lane만 실측인 케이스들
    if (!lane1_has_points && lane2_has_points && !lane3_has_points && !lane4_has_points) {
        if (should_log) std::cout << "[LaneProcessor] Warning: Only Lane2 has points, using previous polyfits" << std::endl;
    }
    if (!lane1_has_points && !lane2_has_points && lane3_has_points && !lane4_has_points) {
        if (should_log) std::cout << "[LaneProcessor] Warning: Only Lane3 has points, using previous polyfits" << std::endl;
    }

    // Lane ordering 체크 (warning만 출력, 롤백은 하지 않음 - 에러 처리는 HasLaneDetectionError에서 수행)
    EnforceLaneOrdering(should_log);

    // Driveway 판단 전에 이전 값 저장
    int saved_prev_driveway = current_driveway_;

    DetermineCurrentDriveway();

    // Driveway 변경 시 outer lane 재구성
    prev_driveway_ = saved_prev_driveway;
    RebuildOuterLanesOnDrivewayChange(should_log);
    prev_driveway_ = current_driveway_;  // 다음 프레임을 위해 업데이트

    GenerateEgoCenterLane(should_log);
}

// ========== 이전 polyfit 저장 ==========
void LaneProcessor::SavePreviousPolyfits() {
    prev_lane1_polyfit_ = lane1_polyfit_;
    prev_lane2_polyfit_ = lane2_polyfit_;
    prev_lane3_polyfit_ = lane3_polyfit_;
    prev_lane4_polyfit_ = lane4_polyfit_;
}

// ========== Lane ordering 안전장치 (warning만 출력, 롤백 제거) ==========
bool LaneProcessor::EnforceLaneOrdering(bool should_log) {
    double y1 = lane1_polyfit_.a0;
    double y2 = lane2_polyfit_.a0;
    double y3 = lane3_polyfit_.a0;
    double y4 = lane4_polyfit_.a0;

    bool ordering_ok = (y1 > y2) && (y2 > y3) && (y3 > y4);

    if (!ordering_ok) {
        if (should_log) {
            std::cout << "[LaneProcessor] WARNING: Lane ordering violated! "
                      << "y1=" << y1 << " y2=" << y2 << " y3=" << y3 << " y4=" << y4
                      << std::endl;
        }
        return false;
    }

    return true;
}

// ========== 에러 감지 함수 ==========
bool LaneProcessor::HasLaneDetectionError(bool should_log) {
    bool has_error = false;

    // 1) 차선 순서 오류 체크: y1 > y2 > y3 > y4 이어야 함
    double y1 = lane1_polyfit_.a0;
    double y2 = lane2_polyfit_.a0;
    double y3 = lane3_polyfit_.a0;
    double y4 = lane4_polyfit_.a0;

    bool ordering_ok = (y1 > y2) && (y2 > y3) && (y3 > y4);
    if (!ordering_ok) {
        has_error = true;
        if (should_log) {
            std::cout << "[LaneProcessor] ERROR: Lane ordering violated! "
                      << "y1=" << y1 << " y2=" << y2 << " y3=" << y3 << " y4=" << y4
                      << std::endl;
        }
    }

    // 2) 차선 폭 이상 체크 (lane2와 lane3 사이)
    double eval_y2 = EvaluatePolynomial(lane2_polyfit_, 0.0);
    double eval_y3 = EvaluatePolynomial(lane3_polyfit_, 0.0);
    double width23 = std::fabs(eval_y2 - eval_y3);

    if (width23 < MIN_LANE_WIDTH || width23 > MAX_LANE_WIDTH) {
        has_error = true;
        if (should_log) {
            std::cout << "[LaneProcessor] ERROR: Lane width out of range! "
                      << "width23=" << width23
                      << " (valid range: " << MIN_LANE_WIDTH << " ~ " << MAX_LANE_WIDTH << ")"
                      << std::endl;
        }
    }

    // 3) polyfit 계수 NaN/Inf 체크
    auto check_finite = [](const interface::PolyfitLane& poly, const std::string& name, bool should_log) {
        bool ok = std::isfinite(poly.a0) && std::isfinite(poly.a1) &&
                  std::isfinite(poly.a2) && std::isfinite(poly.a3);
        if (!ok && should_log) {
            std::cout << "[LaneProcessor] ERROR: " << name << " has NaN/Inf coefficients! "
                      << "a0=" << poly.a0 << " a1=" << poly.a1
                      << " a2=" << poly.a2 << " a3=" << poly.a3 << std::endl;
        }
        return ok;
    };

    if (!check_finite(lane1_polyfit_, "lane1", should_log)) has_error = true;
    if (!check_finite(lane2_polyfit_, "lane2", should_log)) has_error = true;
    if (!check_finite(lane3_polyfit_, "lane3", should_log)) has_error = true;
    if (!check_finite(lane4_polyfit_, "lane4", should_log)) has_error = true;

    return has_error;
}

// ========== HardReset: 전체 메모리 초기화 및 RECOVERY 모드 진입 ==========
void LaneProcessor::HardReset() {
    std::cout << "[LaneProcessor] === HardReset called ===" << std::endl;

    // lane 포인트 메모리 초기화
    lane1_points_.clear();
    lane2_points_.clear();
    lane3_points_.clear();
    lane4_points_.clear();

    prev_lane1_points_.clear();
    prev_lane2_points_.clear();
    prev_lane3_points_.clear();
    prev_lane4_points_.clear();

    // polyfit 초기화
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

    // generated 플래그 초기화
    lane1_is_generated_ = false;
    lane2_is_generated_ = false;
    lane3_is_generated_ = false;
    lane4_is_generated_ = false;

    // driveway 초기화
    current_driveway_ = 2;
    prev_driveway_ = 2;

    // RECOVERY 관련 멤버 초기화
    virtual_left_lane_ = interface::PolyfitLane();
    virtual_right_lane_ = interface::PolyfitLane();
    virtual_left_valid_ = false;
    virtual_right_valid_ = false;
    recovery_frames_ = 0;
    recovery_stable_count_ = 0;

    recovery_left_points_.clear();
    recovery_right_points_.clear();

    // 프레임 카운터 초기화
    total_processed_frames_ = 0;

    // 상태 전환
    tracking_state_ = LaneTrackingState::RECOVERY;
}

// ========== Driveway 판단 (robust version) ==========
void LaneProcessor::DetermineCurrentDriveway() {
    double y1 = lane1_polyfit_.a0;
    double y2 = lane2_polyfit_.a0;
    double y3 = lane3_polyfit_.a0;
    double y4 = lane4_polyfit_.a0;

    struct PairInfo {
        int left_lane;
        int right_lane;
        double y_left;
        double y_right;
        int driveway_id;
    };

    std::vector<PairInfo> pairs = {
        {1, 2, y1, y2, 1},
        {2, 3, y2, y3, 2},
        {3, 4, y3, y4, 3}
    };

    int best_driveway = current_driveway_;
    double best_width = std::numeric_limits<double>::max();

    for (const auto& p : pairs) {
        if ((p.y_left >= 0.0 && p.y_right <= 0.0) ||
            (p.y_left <= 0.0 && p.y_right >= 0.0)) {
            double width = std::fabs(p.y_left - p.y_right);
            if (width < best_width) {
                best_width = width;
                best_driveway = p.driveway_id;
            }
        }
    }

    if (best_width < std::numeric_limits<double>::max()) {
        current_driveway_ = best_driveway;
    }
}

// ========== Ego center lane 생성 ==========
void LaneProcessor::GenerateEgoCenterLane(bool should_log) {
    (void)should_log;
    ego_center_lane_.id = "ego_center";
    ego_center_lane_.frame_id = "vehicle";

    if (current_driveway_ == 1) {
        ego_center_lane_.a0 = (lane1_polyfit_.a0 + lane2_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane1_polyfit_.a1 + lane2_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane1_polyfit_.a2 + lane2_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane1_polyfit_.a3 + lane2_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start = std::max(lane1_polyfit_.x_start, lane2_polyfit_.x_start);
        ego_center_lane_.x_end = std::min(lane1_polyfit_.x_end, lane2_polyfit_.x_end);
    } else if (current_driveway_ == 2) {
        ego_center_lane_.a0 = (lane2_polyfit_.a0 + lane3_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane2_polyfit_.a1 + lane3_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane2_polyfit_.a2 + lane3_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane2_polyfit_.a3 + lane3_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start = std::max(lane2_polyfit_.x_start, lane3_polyfit_.x_start);
        ego_center_lane_.x_end = std::min(lane2_polyfit_.x_end, lane3_polyfit_.x_end);
    } else if (current_driveway_ == 3) {
        ego_center_lane_.a0 = (lane3_polyfit_.a0 + lane4_polyfit_.a0) / 2.0;
        ego_center_lane_.a1 = (lane3_polyfit_.a1 + lane4_polyfit_.a1) / 2.0;
        ego_center_lane_.a2 = (lane3_polyfit_.a2 + lane4_polyfit_.a2) / 2.0;
        ego_center_lane_.a3 = (lane3_polyfit_.a3 + lane4_polyfit_.a3) / 2.0;
        ego_center_lane_.x_start = std::max(lane3_polyfit_.x_start, lane4_polyfit_.x_start);
        ego_center_lane_.x_end = std::min(lane3_polyfit_.x_end, lane4_polyfit_.x_end);
    }
}

// ========== Helper Functions ==========

void LaneProcessor::DownsampleLanePoints() {
    // 포인트 수가 MAX_POINTS_PER_LANE의 1.5배를 넘을 때만 다운샘플 수행
    const double DOWNSAMPLE_TRIGGER_FACTOR = 1.5;

    auto downsample = [DOWNSAMPLE_TRIGGER_FACTOR](std::vector<interface::Point2D>& lane_points) {
        // 충분히 많을 때만 압축
        if (lane_points.size() <= static_cast<size_t>(MAX_POINTS_PER_LANE * DOWNSAMPLE_TRIGGER_FACTOR)) {
            return;
        }

        // x 기준 정렬
        std::sort(lane_points.begin(), lane_points.end(),
                  [](const interface::Point2D& a, const interface::Point2D& b) {
                      return a.x < b.x;
                  });

        std::vector<interface::Point2D> downsampled;
        downsampled.reserve(MAX_POINTS_PER_LANE);

        size_t total = lane_points.size();
        for (size_t i = 0; i < MAX_POINTS_PER_LANE; ++i) {
            size_t idx = (total == 0) ? 0 : (i * total) / MAX_POINTS_PER_LANE;
            if (idx < total) {
                downsampled.push_back(lane_points[idx]);
            }
        }

        lane_points = std::move(downsampled);
    };

    downsample(lane1_points_);
    downsample(lane2_points_);
    downsample(lane3_points_);
    downsample(lane4_points_);
}

void LaneProcessor::FilterStoredPointsByRange(const interface::VehicleState& vehicle_state) {
    (void)vehicle_state;

    // y 범위 필터 상수 (도로 폭에 맞게 조정)
    // NOTE: driveway 변경시 outer lane 메모리를 RebuildOuterLanesOnDrivewayChange()에서
    //       강제로 clear하기 때문에, 여기서는 장기적인 메모리만 제거한다.
    const double MAX_LATERAL_MEMORY = 6.0;

    auto filter = [this, MAX_LATERAL_MEMORY](std::vector<interface::Point2D>& lane_points) {
        lane_points.erase(
            std::remove_if(lane_points.begin(), lane_points.end(),
                [this, MAX_LATERAL_MEMORY](const interface::Point2D& vehicle_pt) {
                    return vehicle_pt.x < MEMORY_X_MIN || vehicle_pt.x > MEMORY_X_MAX ||
                           vehicle_pt.y < -MAX_LATERAL_MEMORY || vehicle_pt.y > MAX_LATERAL_MEMORY;
                }),
            lane_points.end()
        );
    };

    filter(lane1_points_);
    filter(lane2_points_);
    filter(lane3_points_);
    filter(lane4_points_);
}

double LaneProcessor::EvaluatePolynomial(const interface::PolyfitLane& polyfit, double x) const {
    return polyfit.a0 + polyfit.a1 * x + polyfit.a2 * x * x + polyfit.a3 * x * x * x;
}

interface::Point2D LaneProcessor::VehicleToGlobalFrame(
    const interface::Point2D& vehicle_point,
    const interface::VehicleState& vehicle_state) const {

    const double cos_yaw = std::cos(vehicle_state.yaw);
    const double sin_yaw = std::sin(vehicle_state.yaw);

    interface::Point2D global_point;
    global_point.x = vehicle_state.x + cos_yaw * vehicle_point.x - sin_yaw * vehicle_point.y;
    global_point.y = vehicle_state.y + sin_yaw * vehicle_point.x + cos_yaw * vehicle_point.y;

    return global_point;
}

interface::Point2D LaneProcessor::GlobalToVehicleFrame(
    const interface::Point2D& global_point,
    const interface::VehicleState& vehicle_state) const {

    const double dx = global_point.x - vehicle_state.x;
    const double dy = global_point.y - vehicle_state.y;

    const double cos_yaw = std::cos(-vehicle_state.yaw);
    const double sin_yaw = std::sin(-vehicle_state.yaw);

    interface::Point2D vehicle_point;
    vehicle_point.x = cos_yaw * dx - sin_yaw * dy;
    vehicle_point.y = sin_yaw * dx + cos_yaw * dy;

    return vehicle_point;
}

bool LaneProcessor::FitCubicPolynomial(
    const std::vector<interface::Point2D>& points,
    interface::PolyfitLane& out_polyfit) const {

    if (points.size() < 4) {
        return false;
    }

    const size_t n = points.size();
    Eigen::MatrixXd A(n, 4);
    Eigen::VectorXd b(n);

    double x_min = std::numeric_limits<double>::max();
    double x_max = std::numeric_limits<double>::lowest();

    for (size_t i = 0; i < n; ++i) {
        const double x = points[i].x;
        const double y = points[i].y;

        A(i, 0) = 1.0;
        A(i, 1) = x;
        A(i, 2) = x * x;
        A(i, 3) = x * x * x;
        b(i) = y;

        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
    }

    Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    out_polyfit.a0 = coeffs(0);
    out_polyfit.a1 = coeffs(1);
    out_polyfit.a2 = coeffs(2);
    out_polyfit.a3 = coeffs(3);
    out_polyfit.x_start = x_min;
    out_polyfit.x_end = x_max;

    return true;
}

// ========== Getter Functions ==========

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

int LaneProcessor::GetCurrentDriveway() const {
    return current_driveway_;
}

std::vector<geometry_msgs::msg::Point> LaneProcessor::getLane1PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {
    (void)vehicle_state;
    std::vector<geometry_msgs::msg::Point> result;

    for (const auto& vehicle_pt : lane1_points_) {
        geometry_msgs::msg::Point point;
        point.x = vehicle_pt.x;
        point.y = vehicle_pt.y;
        point.z = 0.0;
        result.push_back(point);
    }

    return result;
}

std::vector<geometry_msgs::msg::Point> LaneProcessor::getLane2PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {
    (void)vehicle_state;
    std::vector<geometry_msgs::msg::Point> result;

    for (const auto& vehicle_pt : lane2_points_) {
        geometry_msgs::msg::Point point;
        point.x = vehicle_pt.x;
        point.y = vehicle_pt.y;
        point.z = 0.0;
        result.push_back(point);
    }

    return result;
}

std::vector<geometry_msgs::msg::Point> LaneProcessor::getLane3PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {
    (void)vehicle_state;
    std::vector<geometry_msgs::msg::Point> result;

    for (const auto& vehicle_pt : lane3_points_) {
        geometry_msgs::msg::Point point;
        point.x = vehicle_pt.x;
        point.y = vehicle_pt.y;
        point.z = 0.0;
        result.push_back(point);
    }

    return result;
}

std::vector<geometry_msgs::msg::Point> LaneProcessor::getLane4PointsMemoryVehicleFrame(
    const interface::VehicleState& vehicle_state) const {
    (void)vehicle_state;
    std::vector<geometry_msgs::msg::Point> result;

    for (const auto& vehicle_pt : lane4_points_) {
        geometry_msgs::msg::Point point;
        point.x = vehicle_pt.x;
        point.y = vehicle_pt.y;
        point.z = 0.0;
        result.push_back(point);
    }

    return result;
}

}  // namespace lane_processor
