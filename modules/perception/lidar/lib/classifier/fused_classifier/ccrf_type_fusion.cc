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
#include "modules/perception/lidar/lib/classifier/fused_classifier/ccrf_type_fusion.h"

#include <limits>

#include "cyber/common/file.h"
#include "cyber/common/log.h"
#include "modules/perception/base/object_types.h"
#include "modules/perception/base/point_cloud.h"
#include "modules/perception/lib/config_manager/config_manager.h"
#include "modules/perception/proto/ccrf_type_fusion_config.pb.h"

namespace apollo {
namespace perception {
namespace lidar {

using apollo::common::EigenMap;
using apollo::common::EigenVector;

using ObjectPtr = std::shared_ptr<apollo::perception::base::Object>;
using apollo::cyber::common::GetAbsolutePath;
using apollo::perception::base::ObjectType;

bool CCRFOneShotTypeFusion::Init(const TypeFusionInitOption& option) {
  auto config_manager = lib::ConfigManager::Instance();
  const lib::ModelConfig* model_config = nullptr;
  ACHECK(config_manager->GetModelConfig(Name(), &model_config));
  const std::string work_root = config_manager->work_root();
  std::string config_file;
  std::string root_path;
  ACHECK(model_config->get_value("root_path", &root_path));
  config_file = GetAbsolutePath(work_root, root_path);
  config_file = GetAbsolutePath(config_file, "ccrf_type_fusion.conf");
  CcrfTypeFusionConfig config;
  ACHECK(cyber::common::GetProtoFromFile(config_file, &config));


  // classifiers_property_file_path: "./data/perception/lidar/models/fused_classifier/classifiers.property"

  std::string classifiers_property_file_path =
      GetAbsolutePath(work_root, config.classifiers_property_file_path());
  ACHECK(util::LoadMultipleMatricesFile(classifiers_property_file_path,
                                        &smooth_matrices_));

  //   3

  // DecisionForestClassifier
  // 0.7751 0.0298 0.0639 0.1312
  // 0.2510 0.6802 0.0615 0.0073
  // 0.1904 0.0628 0.6314 0.1155
  // 0.1054 0.0003 0.0038 0.8905

  // CNNSegmentation
  // 0.9095 0.0238 0.0190 0.0476
  // 0.3673 0.5672 0.0642 0.0014
  // 0.1314 0.0078 0.7627 0.0980
  // 0.3383 0.0017 0.0091 0.6508

  // Confidence
  // 1.00 0.00 0.00 0.00
  // 0.40 0.60 0.00 0.00
  // 0.40 0.00 0.60 0.00
  // 0.50 0.00 0.00 0.50

  for (auto& pair : smooth_matrices_) {
    util::NormalizeRow(&pair.second);
    pair.second.transposeInPlace();
    AINFO << "Source: " << pair.first;
    AINFO << std::endl << pair.second;
  }

  confidence_smooth_matrix_ = Matrixd::Identity();
  auto iter = smooth_matrices_.find("Confidence");
  if (iter != smooth_matrices_.end()) {
    confidence_smooth_matrix_ = iter->second;
    smooth_matrices_.erase(iter);
  }
  AINFO << "Confidence: ";
  AINFO << std::endl << confidence_smooth_matrix_;
  // Confidence
  // 1.00 0.00 0.00 0.00
  // 0.40 0.60 0.00 0.00
  // 0.40 0.00 0.60 0.00
  // 0.50 0.00 0.00 0.50

  return true;
}

bool CCRFOneShotTypeFusion::TypeFusion(const TypeFusionOption& option,
                                       ObjectPtr object) {
  if (object == nullptr) {
    return false;
  }
  Vectord log_prob;
  if (!FuseOneShotTypeProbs(object, &log_prob)) {
    return false;
  }
  // exp(probs)
  util::ToExp(&log_prob);
  // 归一化，除以总和
  util::Normalize(&log_prob);
  // matrix转成vector
  util::FromEigenToVector(log_prob, &(object->type_probs));
  // 最大值prob为当前类型
  object->type = static_cast<ObjectType>(std::distance(
      object->type_probs.begin(),
      std::max_element(object->type_probs.begin(), object->type_probs.end())));
  return true;
}

bool CCRFOneShotTypeFusion::FuseOneShotTypeProbs(const ObjectPtr& object,
                                                 Vectord* log_prob) {
  if (object == nullptr) {
    return false;
  }
  if (log_prob == nullptr) {
    return false;
  }

  // 二维数组vector
  const auto& vecs = object->lidar_supplement.raw_probs;
  
  // names只有PointPillarsDetection
  const auto& names = object->lidar_supplement.raw_classification_methods;
  if (vecs.empty()) {
    return false;
  }

  log_prob->setZero();

  Vectord single_prob;
  static const Vectord epsilon = Vectord::Ones() * 1e-6;
  float conf = object->confidence;
  for (size_t i = 0; i < vecs.size(); ++i) {
    auto& vec = vecs[i];
    util::FromStdToVector(vec, &single_prob);
    auto iter = smooth_matrices_.find(names[i]);
    if (vecs.size() == 1 || iter == smooth_matrices_.end()) {
      single_prob = single_prob + epsilon;
    } else {
      single_prob = iter->second * single_prob + epsilon;
    }
    // 归一化，除以总和
    util::Normalize(&single_prob);
    // p(c|x) = p(c|x,o)p(o|x)+ p(c|x,~o)p(~o|x)
    single_prob = conf * single_prob +
                  (1.0 - conf) * confidence_smooth_matrix_ * single_prob;
    // 求log对数
    util::ToLog(&single_prob);
    // 总和
    *log_prob += single_prob;
  }

  return true;
}

bool CCRFSequenceTypeFusion::Init(const TypeFusionInitOption& option) {
  ACHECK(one_shot_fuser_.Init(option));
  auto config_manager = lib::ConfigManager::Instance();
  const lib::ModelConfig* model_config = nullptr;
  ACHECK(config_manager->GetModelConfig(Name(), &model_config));
  const std::string work_root = config_manager->work_root();
  std::string config_file;
  std::string root_path;
  ACHECK(model_config->get_value("root_path", &root_path));
  config_file = GetAbsolutePath(work_root, root_path);
  config_file = GetAbsolutePath(config_file, "ccrf_type_fusion.conf");
  CcrfTypeFusionConfig config;
  ACHECK(cyber::common::GetProtoFromFile(config_file, &config));
  std::string transition_property_file_path =
      GetAbsolutePath(work_root, config.transition_property_file_path());

  // transition_property_file_path: "./data/perception/lidar/models/fused_classifier/transition.property"
  // transition_matrix_alpha: 1.8

  s_alpha_ = config.transition_matrix_alpha();
  ACHECK(util::LoadSingleMatrixFile(transition_property_file_path,
                                    &transition_matrix_));
  transition_matrix_ += Matrixd::Ones() * 1e-6;
  util::NormalizeRow(&transition_matrix_);
  AINFO << "transition matrix";
  AINFO << std::endl << transition_matrix_;
  for (std::size_t i = 0; i < VALID_OBJECT_TYPE; ++i) {
    for (std::size_t j = 0; j < VALID_OBJECT_TYPE; ++j) {
      transition_matrix_(i, j) = log(transition_matrix_(i, j));
    }
  }
  AINFO << std::endl << transition_matrix_;

  // 0.34 0.22 0.33 0.11
  // 0.03 0.90 0.05 0.02
  // 0.03 0.05 0.90 0.02
  // 0.06 0.01 0.03 0.90
  
  return true;
}

bool CCRFSequenceTypeFusion::TypeFusion(const TypeFusionOption& option,
                                        TrackedObjects* tracked_objects) {
  if (tracked_objects == nullptr) {
    return false;
  }
  if (tracked_objects->empty()) {
    return false;
  }
  return FuseWithConditionalProbabilityInference(tracked_objects);
}

bool CCRFSequenceTypeFusion::FuseWithConditionalProbabilityInference(
    TrackedObjects* tracked_objects) {
  // AINFO << "Enter fuse with conditional probability inference";
  fused_oneshot_probs_.resize(tracked_objects->size());

  std::size_t i = 0;
  for (auto& pair : *tracked_objects) {
    ObjectPtr& object = pair.second;
    if (!one_shot_fuser_.FuseOneShotTypeProbs(object,
                                              &fused_oneshot_probs_[i++])) {
      AERROR << "Failed to fuse one short probs in sequence.";
      return false;
    }
  }

  // 维特比算法
  // Use viterbi algorithm to infer the state
  std::size_t length = tracked_objects->size();
  fused_sequence_probs_.resize(length);
  state_back_trace_.resize(length);

  fused_sequence_probs_[0] = fused_oneshot_probs_[0];
  // Add priori knowledge to suppress the sudden-appeared object types.
  fused_sequence_probs_[0] += transition_matrix_.row(0).transpose();

  for (std::size_t i = 1; i < length; ++i) {
    for (std::size_t right = 0; right < VALID_OBJECT_TYPE; ++right) {
      double prob = 0.0;
      double max_prob = -std::numeric_limits<double>::max();
      std::size_t id = 0;
      for (std::size_t left = 0; left < VALID_OBJECT_TYPE; ++left) {
        prob = fused_sequence_probs_[i - 1](left) +
               transition_matrix_(left, right) * s_alpha_ +
               fused_oneshot_probs_[i](right);
        if (prob > max_prob) {
          max_prob = prob;
          id = left;
        }
      }
      fused_sequence_probs_[i](right) = max_prob;
      state_back_trace_[i](right) = static_cast<int>(id);
    }
  }
  ObjectPtr object = tracked_objects->rbegin()->second;
  RecoverFromLogProbability(&fused_sequence_probs_.back(), &object->type_probs,
                            &object->type);
  return true;
}

bool CCRFSequenceTypeFusion::RecoverFromLogProbability(Vectord* prob,
                                                       std::vector<float>* dst,
                                                       ObjectType* type) {
  util::ToExpStable(prob);
  util::Normalize(prob);
  util::FromEigenToVector(*prob, dst);
  *type = static_cast<ObjectType>(
      std::distance(dst->begin(), std::max_element(dst->begin(), dst->end())));
  return true;
}

PERCEPTION_REGISTER_ONESHOTTYPEFUSION(CCRFOneShotTypeFusion);
PERCEPTION_REGISTER_SEQUENCETYPEFUSION(CCRFSequenceTypeFusion);

}  // namespace lidar
}  // namespace perception
}  // namespace apollo
