/******************************************************************************
 * Copyright 2018 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/
#include "modules/perception/fusion/common/kalman_filter.h"
#include "cyber/common/log.h"

namespace apollo {
namespace perception {
namespace fusion {

KalmanFilter::KalmanFilter() : BaseFilter("KalmanFilter") {}

bool KalmanFilter::Init(const Eigen::VectorXd &initial_belief_states,
                        const Eigen::MatrixXd &initial_uncertainty) {
  if (initial_uncertainty.rows() != initial_uncertainty.cols()) {
    AERROR << "the cols and rows of uncertainty martix should be equal";
    return false;
  }
  states_num_ = static_cast<int>(initial_uncertainty.rows());

  if (states_num_ <= 0) {
    AERROR << "state_num should be greater than zero";
    return false;
  }

  if (states_num_ != initial_belief_states.rows()) {
    AERROR << "the rows of state should be equal to state_num";
    return false;
  }

  global_states_ = initial_belief_states;
  global_uncertainty_ = initial_uncertainty;
  // 更新之前的状态值X
  prior_global_states_ = global_states_;

  // 状态预测矩阵F
  transform_matrix_.setIdentity(states_num_, states_num_);
  // 测量噪声
  cur_observation_.setZero(states_num_, 1);
  // 测量噪声协方差矩阵R
  cur_observation_uncertainty_.setIdentity(states_num_, states_num_);

  // 状态转移矩阵H
  c_matrix_.setIdentity(states_num_, states_num_);
  // 预测噪声协方差矩阵Q
  env_uncertainty_.setZero(states_num_, states_num_);

  gain_break_down_.setZero(states_num_, 1);
  value_break_down_.setZero(states_num_, 1);

  // K值
  kalman_gain_.setZero(states_num_, states_num_);
  init_ = true;
  return true;
}

bool KalmanFilter::Predict(const Eigen::MatrixXd &transform_matrix,
                           const Eigen::MatrixXd &env_uncertainty_matrix) {
  if (!init_) {
    AERROR << "Predict: Kalman Filter initialize not successfully";
    return false;
  }
  if (transform_matrix.rows() != states_num_) {
    AERROR << "the rows of transform matrix should be equal to state_num";
    return false;
  }
  if (transform_matrix.cols() != states_num_) {
    AERROR << "the cols of transform matrix should be equal to state_num";
    return false;
  }
  if (env_uncertainty_matrix.rows() != states_num_) {
    AERROR << "the rows of env uncertainty should be equal to state_num";
    return false;
  }
  if (env_uncertainty_matrix.cols() != states_num_) {
    AERROR << "the cols of env uncertainty should be equal to state_num";
    return false;
  }
  // 状态值X
  transform_matrix_ = transform_matrix;
  // 预测噪声协方差矩阵Q
  env_uncertainty_ = env_uncertainty_matrix;

  // X_ = F * X
  global_states_ = transform_matrix_ * global_states_;
  // P_ = F * P * F_t + Q
  global_uncertainty_ =
      transform_matrix_ * global_uncertainty_ * transform_matrix_.transpose() +
      env_uncertainty_;
  return true;
}

bool KalmanFilter::Correct(const Eigen::VectorXd &cur_observation,
                           const Eigen::MatrixXd &cur_observation_uncertainty) {
  if (!init_) {
    AERROR << "Correct: Kalman Filter initialize not successfully";
    return false;
  }
  if (cur_observation.rows() != states_num_) {
    AERROR << "the rows of current observation should be equal to state_num";
    return false;
  }
  if (cur_observation_uncertainty.rows() != states_num_) {
    AERROR << "the rows of current observation uncertainty "
              "should be equal to state_num";
    return false;
  }
  if (cur_observation_uncertainty.cols() != states_num_) {
    AERROR << "the cols of current observation uncertainty "
              "should be equal to state_num";
    return false;
  }

  // 观测值Z
  cur_observation_ = cur_observation;
  // 观测噪声协方差矩阵R
  cur_observation_uncertainty_ = cur_observation_uncertainty;
  // K = P_ * H_t * ( H * P_ * H_t + R )^-1
  kalman_gain_ = global_uncertainty_ * c_matrix_.transpose() *
                 (c_matrix_ * global_uncertainty_ * c_matrix_.transpose() +
                  cur_observation_uncertainty_)
                     .inverse();
  // X = X_ + K * ( Z - H * X_ )
  global_states_ = global_states_ + kalman_gain_ * (cur_observation_ -
                                                    c_matrix_ * global_states_);
  Eigen::MatrixXd tmp_identity;
  tmp_identity.setIdentity(states_num_, states_num_);

  // note:Apollo使用了非最优卡尔曼增益的估计误差协方差矩阵更新公式
  // 原因参考：https://mp.weixin.qq.com/s/mErKo1CTV14iY2cKkRUVhA，https://zhuanlan.zhihu.com/p/33852112

  // P = (I - K * H) * P_ * (I - K * H)_t + K * R * K_t
  global_uncertainty_ =
      (tmp_identity - kalman_gain_ * c_matrix_) * global_uncertainty_ *
          (tmp_identity - kalman_gain_ * c_matrix_).transpose() +
      kalman_gain_ * cur_observation_uncertainty_ * kalman_gain_.transpose();
  return true;
}

bool KalmanFilter::SetControlMatrix(const Eigen::MatrixXd &control_matrix) {
  if (!init_) {
    AERROR << "SetControlMatrix: Kalman Filter initialize not successfully";
    return false;
  }
  if (control_matrix.rows() != states_num_ ||
      control_matrix.cols() != states_num_) {
    AERROR << "the rows/cols of control matrix should be equal to state_num";
    return false;
  }
  // 设置状态转移矩阵Ｈ
  c_matrix_ = control_matrix;
  return true;
}

Eigen::VectorXd KalmanFilter::GetStates() const { return global_states_; }

Eigen::MatrixXd KalmanFilter::GetUncertainty() const {
  return global_uncertainty_;
}

bool KalmanFilter::SetGainBreakdownThresh(const std::vector<bool> &break_down,
                                          const float threshold) {
  if (static_cast<int>(break_down.size()) != states_num_) {
    return false;
  }
  for (int i = 0; i < states_num_; i++) {
    if (break_down[i]) {
      gain_break_down_(i) = 1;
    }
  }
  gain_break_down_threshold_ = threshold;
  return true;
}

bool KalmanFilter::SetValueBreakdownThresh(const std::vector<bool> &break_down,
                                           const float threshold) {
  if (static_cast<int>(break_down.size()) != states_num_) {
    return false;
  }
  for (int i = 0; i < states_num_; i++) {
    if (break_down[i]) {
      value_break_down_(i) = 1;
    }
  }
  value_break_down_threshold_ = threshold;
  return true;
}

// 修正过大的加速度增益和较小的速度噪声
// 其实就是定义的简单逻辑：加速度增益大于2时，减小一点；速度小于0.05时，认为是噪声，设为0
void KalmanFilter::CorrectionBreakdown() {
  // 更新前后的差值
  Eigen::VectorXd states_gain = global_states_ - prior_global_states_;
  // delta_ax,delta_ay
  Eigen::VectorXd breakdown_diff = states_gain.cwiseProduct(gain_break_down_);
  // X先减去加速度差值
  global_states_ -= breakdown_diff;
  // delta_a大于阈值2
  if (breakdown_diff.norm() > gain_break_down_threshold_) {
    // value/norm()
    breakdown_diff.normalize();
    breakdown_diff *= gain_break_down_threshold_;
  }
  // X加上修改后的加速度差值
  // x,y,vx,vy,(ax0+2*delta_ax/delta_a),(ay0+2*delta_ay/delta_a)
  global_states_ += breakdown_diff;

  Eigen::VectorXd temp;
  temp.setOnes(states_num_, 1);
  // v小于阈值0.05，v=0
  if ((global_states_.cwiseProduct(value_break_down_)).norm() <
      value_break_down_threshold_) {
    global_states_ = global_states_.cwiseProduct(temp - value_break_down_);
  }
  prior_global_states_ = global_states_;
}

bool KalmanFilter::DeCorrelation(int x, int y, int x_len, int y_len) {
  if (x >= states_num_ || y >= states_num_ || x + x_len >= states_num_ ||
      y + y_len >= states_num_) {
    return false;
  }
  for (int i = 0; i < x_len; i++) {
    for (int j = 0; j < y_len; j++) {
      global_uncertainty_(x + i, y + j) = 0;
    }
  }
  return true;
}

}  // namespace fusion
}  // namespace perception
}  // namespace apollo
