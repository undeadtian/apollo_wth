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
#pragma once

#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace apollo {
namespace perception {
namespace fusion {

struct DstCommonData {
  // ensure initialize DSTEvidence once
  bool init_ = false;
  // 假设空间集合location，大小？
  size_t fod_loc_ = 0;
  // 假设空间集合
  std::vector<uint64_t> fod_subsets_;

  // 假设空间集合基数？
  // for transforming to probability effectively
  std::vector<size_t> fod_subset_cardinalities_;

  // 假设空间集合对应的名字
  std::vector<std::string> fod_subset_names_;

  // 组合关系对，二维数组
  // for combining two bbas effectively.
  std::vector<std::vector<std::pair<size_t, size_t>>> combination_relations_;

  // 假设空间关系，二维数组
  // for computing support vector effectively
  std::vector<std::vector<size_t>> subset_relations_;

  // 内在关系，二维数组
  // for computing plausibility vector effectively
  std::vector<std::vector<size_t>> inter_relations_;

  // 假设空间对应index的map
  std::map<uint64_t, size_t> subsets_ind_map_;
};

typedef DstCommonData* DstCommonDataPtr;

// @brief: A singleton class to mange the set of fod subset and the
// intersection relationship between them.
class DstManager {
 public:

  // 初始化实例
  static DstManager* Instance() {
    static DstManager dst_manager;
    return &dst_manager;
  }
  // brief: app initialization
  // param [in]: app_name
  // param [in]: fod_subsets, hypotheses sets
  // param [in]: fod_subset_names

  // note：App=Application，可以理解为应用程序的出入口，一般处理启动和退出程序时要读取
  // 和写入的设置信息，和一些全局变量。
  // 添加app，使用假设空间初始化
  bool AddApp(const std::string& app_name,
              const std::vector<uint64_t>& fod_subsets,
              const std::vector<std::string>& fod_subset_names =
                  std::vector<std::string>());
  bool IsAppAdded(const std::string& app_name);

  DstCommonDataPtr GetAppDataPtr(const std::string& app_name);
  // 假设空间 to index
  size_t FodSubsetToInd(const std::string& app_name,
                        const uint64_t& fod_subset);
  // index to 假设空间
  uint64_t IndToFodSubset(const std::string& app_name, const size_t& ind);

 private:
  DstManager() {}

  // 建立 假设空间和index 的map图
  void BuildSubsetsIndMap(DstCommonData* dst_data);
  // fod check, put fod in fod_subsets to ensure BBA's validity after
  // default construction.
  void FodCheck(DstCommonData* dst_data);

  // compute the cardinality of fod_subset which means counting set bits in
  // an integer
  // 计算假设空间的元素技术
  void ComputeCardinalities(DstCommonData* st_data);
  // 计算假设空间的元素之间的交集和子集
  bool ComputeRelations(DstCommonData* dst_data);

  // 赋值名字
  void BuildNamesMap(const std::vector<std::string>& fod_subset_names,
                     DstCommonData* dst_data);

 private:
  // Dst data map
  // 每个app有不同的name，实例化不同的commonData
  std::map<std::string, DstCommonData> dst_common_data_;

  // map锁
  std::mutex map_mutex_;
};

class Dst {
 public:
  explicit Dst(const std::string& app_name);

  // setter设置
  // 直接赋值
  bool SetBbaVec(const std::vector<double>& bba_vec);

  // strictly require the fod in bba_map is valid
  // 用map赋值
  bool SetBba(const std::map<uint64_t, double>& bba_map);

  // 计算：Spt信度函数 Pls似真度函数 Utc信度空间大小
  void ComputeSptPlsUct() const;

  // 计算：所有假设的概率值
  void ComputeProbability() const;

  // getter

  // 获取mass函数
  const std::vector<double>& GetBbaVec() const { return bba_vec_; }
  // 获取mass函数
  const size_t GetBbaSize() const { return bba_vec_.size(); }

  // 获取假设空间对应的index的mass函数：bba_vec_[inx]
  double GetSubsetBfmass(uint64_t fod_subset) const;

  // 获取index的mass函数：bba_vec_[ind]
  double GetIndBfmass(size_t ind) const;

  // 获取Spt信度函数 Pls似真度函数 Utc信度空间大小 所有假设空间的元素的概率值
  const std::vector<double>& GetSupportVec() const { return support_vec_; }
  const std::vector<double>& GetPlausibilityVec() const {
    return plausibility_vec_;
  }
  const std::vector<double>& GetUncertaintyVec() const {
    return uncertainty_vec_;
  }
  const std::vector<double>& GetProbabilityVec() const {
    return probability_vec_;
  }
  std::string PrintBba() const;

  // 重载
  // mass相加，Dempster-Shafer合成公式
  friend Dst operator+(const Dst& lhs, const Dst& rhs);
  // mass乘以w
  friend Dst operator*(const Dst& dst_evidence, double w);

  std::string Name() const { return app_name_; }

 private:
  // mass函数归一化
  void Normalize();
  void SelfCheck() const;

 private:
  std::string app_name_;
  // the construction of following vectors is manual.
  mutable DstCommonDataPtr dst_data_ptr_ = nullptr;
  // mass函数
  mutable std::vector<double> bba_vec_;
  // Spt信度函数 
  mutable std::vector<double> support_vec_;
  // Pls似真度函数 
  mutable std::vector<double> plausibility_vec_;
  // Utc信度空间大小 
  mutable std::vector<double> uncertainty_vec_;
  // 所有假设空间的元素的概率值
  mutable std::vector<double> probability_vec_;
};

}  // namespace fusion
}  // namespace perception
}  // namespace apollo
