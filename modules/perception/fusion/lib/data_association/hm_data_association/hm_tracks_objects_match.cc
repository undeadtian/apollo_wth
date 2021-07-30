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
#include "modules/perception/fusion/lib/data_association/hm_data_association/hm_tracks_objects_match.h"

#include <map>
#include <numeric>
#include <utility>

#include "modules/perception/common/graph/secure_matrix.h"

namespace apollo {
namespace perception {
namespace fusion {

double HMTrackersObjectsAssociation::s_match_distance_thresh_ = 4.0;
double HMTrackersObjectsAssociation::s_match_distance_bound_ = 100.0;
/* this is a slack threshold for camera 2 lidar/radar association.
 * consider the ave 2d-to-3d error is 7%, 30m is 15% of 200m, which
 * is 2 times of ave error around 200m. */
double HMTrackersObjectsAssociation::s_association_center_dist_threshold_ =
    30.0;

template <typename T>
void extract_vector(const std::vector<T>& vec,
                    const std::vector<size_t>& subset_inds,
                    std::vector<T>* sub_vec) {
  sub_vec->reserve(subset_inds.size());
  sub_vec->clear();
  for (auto subset_ind : subset_inds) {
    sub_vec->push_back(vec[subset_ind]);
  }
}

//关联 sensor_objects--fusion_tracks
bool HMTrackersObjectsAssociation::Associate(
    const AssociationOptions& options, SensorFramePtr sensor_measurements,
    ScenePtr scene, AssociationResult* association_result) 
{
  //sensor观测obj
  const std::vector<SensorObjectPtr>& sensor_objects 
    = sensor_measurements->GetForegroundObjects();
  
  //tracks航迹
  const std::vector<TrackPtr>& fusion_tracks = scene->GetForegroundTracks();

  //关联矩阵
  std::vector<std::vector<double>> association_mat;

  //校验，若航迹或观测为空，未分配航迹和未分配观测初始化大小，return不进行assign
  if (fusion_tracks.empty() || sensor_objects.empty()) {
    association_result->unassigned_tracks.resize(fusion_tracks.size());
    association_result->unassigned_measurements.resize(sensor_objects.size());
    // 0,1,2,3...简洁写法初始化
    std::iota(association_result->unassigned_tracks.begin(),
              association_result->unassigned_tracks.end(), 0);
    std::iota(association_result->unassigned_measurements.begin(),
              association_result->unassigned_measurements.end(), 0);
    return true;
  }

  //sensor型号
  std::string measurement_sensor_id = sensor_objects[0]->GetSensorId();
  //sensor时间戳
  double measurement_timestamp = sensor_objects[0]->GetTimestamp();
  //重置之前的计算距离（obj和track）
  track_object_distance_.ResetProjectionCache(measurement_sensor_id,
                                              measurement_timestamp);
  //前毫米波雷达不进行ID匹配                                  
  bool do_nothing = (sensor_objects[0]->GetSensorId() == "radar_front");

  // 1.ID进行匹配,track之前匹配过的障碍物ID，得到association_result的三个结果，
  //   存储的是fusion_tracks和sensor_objects的index
  IdAssign(fusion_tracks, sensor_objects, &association_result->assignments,
           &association_result->unassigned_tracks,
           &association_result->unassigned_measurements, do_nothing, false);

  //4*4的齐次转换矩阵
  Eigen::Affine3d pose;

  //sensor2world的转换
  sensor_measurements->GetPose(&pose);

  //平移向量
  Eigen::Vector3d ref_point = pose.translation();

  ADEBUG << "association_measurement_timestamp@" << measurement_timestamp;

  // 2.计算关联距离矩阵association_mat[num_track,num_meas],取最小欧式距离
  ComputeAssociationDistanceMat(fusion_tracks, sensor_objects, ref_point,
                                association_result->unassigned_tracks,
                                association_result->unassigned_measurements,
                                &association_mat);

  int num_track = static_cast<int>(fusion_tracks.size());
  int num_measurement = static_cast<int>(sensor_objects.size());

  // 初始化置0,航迹track到观测object的距离,观测到航迹的距离
  association_result->track2measurements_dist.assign(num_track, 0);
  association_result->measurement2track_dist.assign(num_measurement, 0);

  // g:global l:local 两个转换便于查询观测和航迹的index
  // track_ind_l2g[i]=track_index, track_ind_g2l[track_index]=i. 
  std::vector<int> track_ind_g2l;
  track_ind_g2l.resize(num_track, -1);
  //track_ind_l2g
  std::vector<size_t> track_ind_l2g = association_result->unassigned_tracks;
  for (size_t i = 0; i < association_result->unassigned_tracks.size(); i++) {
    track_ind_g2l[association_result->unassigned_tracks[i]] =
        static_cast<int>(i);
  }

  // measurement_ind_l2g[i]=obj_index, measurement_ind_g2l[obj_index]=i;
  std::vector<int> measurement_ind_g2l;
  measurement_ind_g2l.resize(num_measurement, -1);
  // measurement_ind_l2g
  std::vector<size_t> measurement_ind_l2g =
      association_result->unassigned_measurements;
  for (size_t i = 0; i < association_result->unassigned_measurements.size();
       i++) {
    measurement_ind_g2l[association_result->unassigned_measurements[i]] =
        static_cast<int>(i);
  }

  //校验，未分配航迹或未分配观测为空，则结束
  if (association_result->unassigned_tracks.empty() ||
      association_result->unassigned_measurements.empty()) {
    return true;
  }

  // 3.最小化匹配(匈牙利匹配)
  // 关联矩阵对应的是i，通过l2g转换为index，插入到对应的三个vector中
  bool state = MinimizeAssignment(
      association_mat, track_ind_l2g, measurement_ind_l2g,
      &association_result->assignments, &association_result->unassigned_tracks,
      &association_result->unassigned_measurements);

  // start do post assign
  // 4.单独再做一遍IdAssign，未分配的航迹（仅有相机观测生成的）和未分配的观测
  std::vector<TrackMeasurmentPair> post_assignments;
  PostIdAssign(fusion_tracks, sensor_objects,
               association_result->unassigned_tracks,
               association_result->unassigned_measurements, &post_assignments);
  association_result->assignments.insert(association_result->assignments.end(),
                                         post_assignments.begin(),
                                         post_assignments.end());

  // 5.排除匹配了的，生成未匹配的航迹track和观测meas
  GenerateUnassignedData(fusion_tracks.size(), sensor_objects.size(),
                         association_result->assignments,
                         &association_result->unassigned_tracks,
                         &association_result->unassigned_measurements);
  // 6.保存航迹track和观测meas的关联值（最小距离），到association_result中
  ComputeDistance(fusion_tracks, sensor_objects,
                  association_result->unassigned_tracks, track_ind_g2l,
                  measurement_ind_g2l, measurement_ind_l2g, association_mat,
                  association_result);

  AINFO << "association: measurement_num = " << sensor_objects.size()
        << ", track_num = " << fusion_tracks.size()
        << ", assignments = " << association_result->assignments.size()
        << ", unassigned_tracks = "
        << association_result->unassigned_tracks.size()
        << ", unassigned_measuremnets = "
        << association_result->unassigned_measurements.size();

  return state;
}
void HMTrackersObjectsAssociation::PostIdAssign(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const std::vector<size_t>& unassigned_fusion_tracks,
    const std::vector<size_t>& unassigned_sensor_objects,
    std::vector<TrackMeasurmentPair>* post_assignments) {
  // 有效未分配的航迹
  std::vector<size_t> valid_unassigned_tracks;
  valid_unassigned_tracks.reserve(unassigned_fusion_tracks.size());
  // only camera track
  // lambda表达式，航迹中只有camera为有效航迹
  auto is_valid_track = [](const TrackPtr& fusion_track) {
    SensorObjectConstPtr camera_obj = fusion_track->GetLatestCameraObject();
    return camera_obj != nullptr &&
           fusion_track->GetLatestLidarObject() == nullptr;
    // && fusion_track->GetLatestRadarObject() == nullptr;
  };

  // 航迹中只有camera为有效航迹ID
  // valid_unassigned_tracks[i]=index
  for (auto unassigned_track_id : unassigned_fusion_tracks) {
    if (is_valid_track(fusion_tracks[unassigned_track_id])) {
      valid_unassigned_tracks.push_back(unassigned_track_id);
    }
  }
  // 根据ID提取对应的航迹和观测
  std::vector<TrackPtr> sub_tracks;
  std::vector<SensorObjectPtr> sub_objects;
  extract_vector(fusion_tracks, valid_unassigned_tracks, &sub_tracks);
  extract_vector(sensor_objects, unassigned_sensor_objects, &sub_objects);

  // 仅有camera的未分配航迹和未分配的观测再做一遍IdAssign
  // TODOtcx:因为之前对窄视角相机的观测直接return了？
  std::vector<size_t> tmp1, tmp2;
  IdAssign(sub_tracks, sub_objects, post_assignments, &tmp1, &tmp2, false,
           true);
  
  // 结果
  for (auto& post_assignment : *post_assignments) {
    post_assignment.first = valid_unassigned_tracks[post_assignment.first];
    post_assignment.second = unassigned_sensor_objects[post_assignment.second];
  }
}

bool HMTrackersObjectsAssociation::MinimizeAssignment(
    const std::vector<std::vector<double>>& association_mat,
    const std::vector<size_t>& track_ind_l2g,
    const std::vector<size_t>& measurement_ind_l2g,
    std::vector<TrackMeasurmentPair>* assignments,
    std::vector<size_t>* unassigned_tracks,
    std::vector<size_t>* unassigned_measurements) {
  //匈牙利匹配:最小距离
  common::GatedHungarianMatcher<float>::OptimizeFlag opt_flag =
      common::GatedHungarianMatcher<float>::OptimizeFlag::OPTMIN;
  //关联矩阵指针 指向匈牙利类的全局关联矩阵
  common::SecureMat<float>* global_costs = optimizer_.mutable_global_costs();
  //行:tracks 列:object(观测)
  int rows = static_cast<int>(unassigned_tracks->size());
  int cols = static_cast<int>(unassigned_measurements->size());

  //最小距离关联矩阵赋值给指针global_costs，即赋值给匈牙利类optimizer_的关联矩阵变量
  global_costs->Resize(rows, cols);
  for (int r_i = 0; r_i < rows; r_i++) {
    for (int c_i = 0; c_i < cols; c_i++) {
      (*global_costs)(r_i, c_i) = static_cast<float>(association_mat[r_i][c_i]);
    }
  }

  // 三个vector存储的都是关联矩阵的序号i，要转换到global的index
  std::vector<TrackMeasurmentPair> local_assignments;
  std::vector<size_t> local_unassigned_tracks;
  std::vector<size_t> local_unassigned_measurements;
  
  //阈值4,边界100，计算连通子图，每个连通子图进行匈牙利匹配
  optimizer_.Match(static_cast<float>(s_match_distance_thresh_),
                   static_cast<float>(s_match_distance_bound_), opt_flag,
                   &local_assignments, &local_unassigned_tracks,
                   &local_unassigned_measurements);

  //在之前Idassign后，补充现在匹配上的航迹和观测的index
  for (auto assign : local_assignments) {
    assignments->push_back(std::make_pair(track_ind_l2g[assign.first],
                                          measurement_ind_l2g[assign.second]));
  }

  // 清除之前未匹配的结果，做了关联计算之后要重新生成新的未匹配结果
  unassigned_tracks->clear();
  unassigned_measurements->clear();
  //未匹配上的track_index
  for (auto un_track : local_unassigned_tracks) {
    unassigned_tracks->push_back(track_ind_l2g[un_track]);
  }
  //未匹配上的观测object_index
  for (auto un_mea : local_unassigned_measurements) {
    unassigned_measurements->push_back(measurement_ind_l2g[un_mea]);
  }
  return true;
}

void HMTrackersObjectsAssociation::ComputeDistance(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const std::vector<size_t>& unassigned_fusion_tracks,
    const std::vector<int>& track_ind_g2l,
    const std::vector<int>& measurement_ind_g2l,
    const std::vector<size_t>& measurement_ind_l2g,
    const std::vector<std::vector<double>>& association_mat,
    AssociationResult* association_result) {
  // 1.track2measurements_dist保存第一遍IdAssign之后的关联矩阵的距离值
  // 当时的未分配航迹和未分配观测生成的关联矩阵
  for (size_t i = 0; i < association_result->assignments.size(); i++) {
    int track_ind = static_cast<int>(association_result->assignments[i].first);
    int measurement_ind =
        static_cast<int>(association_result->assignments[i].second);
    int track_ind_loc = track_ind_g2l[track_ind];
    int measurement_ind_loc = measurement_ind_g2l[measurement_ind];
    if (track_ind_loc >= 0 && measurement_ind_loc >= 0) {
      association_result->track2measurements_dist[track_ind] =
          association_mat[track_ind_loc][measurement_ind_loc];
      association_result->measurement2track_dist[measurement_ind] =
          association_mat[track_ind_loc][measurement_ind_loc];
    }
  }

  // 2.未分配的track，保存对应的track2measurements_dist
  for (size_t i = 0; i < association_result->unassigned_tracks.size(); i++) {
    int track_ind = static_cast<int>(unassigned_fusion_tracks[i]);
    int track_ind_loc = track_ind_g2l[track_ind];
    association_result->track2measurements_dist[track_ind] =
        association_mat[track_ind_loc][0];
    int min_m_loc = 0;
    // 该行（航迹）中找到最小的关联值，min_m_loc等于该列
    for (size_t j = 1; j < association_mat[track_ind_loc].size(); j++) {
      if (association_result->track2measurements_dist[track_ind] >
          association_mat[track_ind_loc][j]) {
        association_result->track2measurements_dist[track_ind] =
            association_mat[track_ind_loc][j];
        min_m_loc = static_cast<int>(j);
      }
    }
    //最小关联值的观测
    int min_m_ind = static_cast<int>(measurement_ind_l2g[min_m_loc]);
    const SensorObjectPtr& min_sensor_object = sensor_objects[min_m_ind];
    const TrackPtr& fusion_track = fusion_tracks[track_ind];、
    SensorObjectConstPtr lidar_object = fusion_track->GetLatestLidarObject();
    SensorObjectConstPtr radar_object = fusion_track->GetLatestRadarObject();

    // TODOtcx:不太懂这里是在算什么
    if (IsCamera(min_sensor_object)) {
      // TODO(linjian) not reasonable,
      // just for return dist score, the dist score is
      // a similarity probability [0, 1] 1 is the best
      association_result->track2measurements_dist[track_ind] = 0.0;
      for (size_t j = 0; j < association_mat[track_ind_loc].size(); ++j) {
        double dist_score = 0.0;
        if (lidar_object != nullptr) {
          dist_score = track_object_distance_.ComputeLidarCameraSimilarity(
              lidar_object, sensor_objects[measurement_ind_l2g[j]],
              IsLidar(sensor_objects[measurement_ind_l2g[j]]));
        } else if (radar_object != nullptr) {
          dist_score = track_object_distance_.ComputeRadarCameraSimilarity(
              radar_object, sensor_objects[measurement_ind_l2g[j]]);
        }
        association_result->track2measurements_dist[track_ind] = std::max(
            association_result->track2measurements_dist[track_ind], dist_score);
      }
    }
  }
  // 3.未分配的meas，保存对应的measurement2track_dist，求关联矩阵该观测的最小值
  for (size_t i = 0; i < association_result->unassigned_measurements.size();
       i++) {
    int m_ind =
        static_cast<int>(association_result->unassigned_measurements[i]);
    int m_ind_loc = measurement_ind_g2l[m_ind];
    association_result->measurement2track_dist[m_ind] =
        association_mat[0][m_ind_loc];
    for (size_t j = 1; j < association_mat.size(); j++) {
      if (association_result->measurement2track_dist[m_ind] >
          association_mat[j][m_ind_loc]) {
        association_result->measurement2track_dist[m_ind] =
            association_mat[j][m_ind_loc];
      }
    }
  }
}

void HMTrackersObjectsAssociation::ComputeAssociationDistanceMat(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    const Eigen::Vector3d& ref_point,
    const std::vector<size_t>& unassigned_tracks,
    const std::vector<size_t>& unassigned_measurements,
    std::vector<std::vector<double>>* association_mat) {
  // if (sensor_objects.empty()) return;
  TrackObjectDistanceOptions opt;
  // TODO(linjian) ref_point
  Eigen::Vector3d tmp = Eigen::Vector3d::Zero();
  opt.ref_point = &tmp;

  //两重for循环,外循环track航迹,内循环object观测
  association_mat->resize(unassigned_tracks.size());
  for (size_t i = 0; i < unassigned_tracks.size(); ++i) {
    int fusion_idx = static_cast<int>(unassigned_tracks[i]);
    const TrackPtr& fusion_track = fusion_tracks[fusion_idx];

    (*association_mat)[i].resize(unassigned_measurements.size());
    for (size_t j = 0; j < unassigned_measurements.size(); ++j) {
      int sensor_idx = static_cast<int>(unassigned_measurements[j]);
      const SensorObjectPtr& sensor_object = sensor_objects[sensor_idx];
      double distance = s_match_distance_thresh_; // =4

      //中心点距离
      double center_dist =
          (sensor_object->GetBaseObject()->center -
           fusion_track->GetFusedObject()->GetBaseObject()->center)
              .norm();
      //中心点距离<30米
      if (center_dist < s_association_center_dist_threshold_) { //30
        //计算欧式距离
        distance =
            track_object_distance_.Compute(fusion_track, sensor_object, opt);
      } else {
        ADEBUG << "center_distance " << center_dist
               << " exceeds slack threshold "
               << s_association_center_dist_threshold_
               << ", track_id: " << fusion_track->GetTrackId()
               << ", obs_id: " << sensor_object->GetBaseObject()->track_id;
      }
      (*association_mat)[i][j] = distance;
      ADEBUG << "track_id: " << fusion_track->GetTrackId()
             << ", obs_id: " << sensor_object->GetBaseObject()->track_id
             << ", distance: " << distance;
    }
  }
}

/**ID匹配,track关联之前匹配过的障碍物ID,主要是lidar和camera,radar直接返回了
 * assignments--track航迹和object观测配对
 * unassigned_fusion_tracks--track里没有该obj
 * unassigned_sensor_objects--obj不在track里
*/
void HMTrackersObjectsAssociation::IdAssign(
    const std::vector<TrackPtr>& fusion_tracks,
    const std::vector<SensorObjectPtr>& sensor_objects,
    std::vector<TrackMeasurmentPair>* assignments,
    std::vector<size_t>* unassigned_fusion_tracks,
    std::vector<size_t>* unassigned_sensor_objects, bool do_nothing,
    bool post) 
{
  size_t num_track = fusion_tracks.size();
  size_t num_obj = sensor_objects.size();

  //校验,初始化,radar是do_nothing
  if (num_track == 0 || num_obj == 0 || do_nothing) {
    unassigned_fusion_tracks->resize(num_track);
    unassigned_sensor_objects->resize(num_obj);
    // 0,1,2...num_track
    std::iota(unassigned_fusion_tracks->begin(),
              unassigned_fusion_tracks->end(), 0);
    // 0,1,2...num_obj
    std::iota(unassigned_sensor_objects->begin(),
              unassigned_sensor_objects->end(), 0);
    return;
  }
  //传感器类型
  const std::string sensor_id = sensor_objects[0]->GetSensorId();

  std::map<int, int> sensor_id_2_track_ind;

  //1.trackers循环,找到track中该传感器对应的object_id，放入到map中
  for (size_t i = 0; i < num_track; i++) {

    //sensor_id对应的track中的object(lidar,radar,camera)
    SensorObjectConstPtr obj = fusion_tracks[i]->GetSensorObject(sensor_id);

    /* when camera system has sub-fusion of obstacle & narrow, they share
     * the same track-id sequence. thus, latest camera object is ok for
     * camera id assign and its information is more up to date. */
    if (sensor_id == "front_6mm" || sensor_id == "front_12mm") {
      obj = fusion_tracks[i]->GetLatestCameraObject();
    }
    if (obj == nullptr) {
      continue;
    }

    //匹配track中obj,[obj_id,track_index]存入到map中
    sensor_id_2_track_ind[obj->GetBaseObject()->track_id] = static_cast<int>(i);
  }

  //保留哪些Index之前是匹配上的  track和object匹配过
  std::vector<bool> fusion_used(num_track, false);
  std::vector<bool> sensor_used(num_obj, false);

  //2.objects循环，在map中寻找当前帧的obj_id，找到了就相互配对，对应位置置true
  for (size_t i = 0; i < num_obj; i++) {

    //object的id
    int track_id = sensor_objects[i]->GetBaseObject()->track_id;

    //找该object对应的map索引
    auto it = sensor_id_2_track_ind.find(track_id);

    // 非post且传感器是窄视角（长焦）相机，不做id_assign。
    // associate函数中，post为false
    // 第一遍IdAssign不对窄视角相机观测做匹配
    // TODOtcx:没搞明白（第二遍PostIdAssign对单独的相机航迹和未分配的观测做匹配）

    // In id_assign, we don't assign the narrow camera object
    // with the track which only have narrow camera object
    // In post id_assign, we do this.
    if (!post && (sensor_id == "front_6mm" || sensor_id == "front_12mm"))
      continue;

    //该object在之前trackers中,相应的位置置true,并进行匹配到相应tracker上,用pair保存
    if (it != sensor_id_2_track_ind.end()) {
      sensor_used[i] = true;
      fusion_used[it->second] = true;
      //航迹和观测配对（第几个航迹和第几个观测，存储的是各自vector的index）
      assignments->push_back(std::make_pair(it->second, i));
    }
  }

  //3.没有ID匹配到的tracker
  for (size_t i = 0; i < fusion_used.size(); ++i) {
    if (!fusion_used[i]) {
      unassigned_fusion_tracks->push_back(i);
    }
  }
  
  //4.没有ID匹配到的object
  for (size_t i = 0; i < sensor_used.size(); ++i) {
    if (!sensor_used[i]) {
      unassigned_sensor_objects->push_back(i);
    }
  }
}

void HMTrackersObjectsAssociation::GenerateUnassignedData(
    size_t track_num, size_t objects_num,
    const std::vector<TrackMeasurmentPair>& assignments,
    std::vector<size_t>* unassigned_tracks,
    std::vector<size_t>* unassigned_objects) {
  
  std::vector<bool> track_flags(track_num, false);
  std::vector<bool> objects_flags(objects_num, false);
  // 匹配了的置true
  for (auto assignment : assignments) {
    track_flags[assignment.first] = true;
    objects_flags[assignment.second] = true;
  }
  // 其他的全是未匹配的
  unassigned_tracks->clear(), unassigned_tracks->reserve(track_num);
  unassigned_objects->clear(), unassigned_objects->reserve(objects_num);
  for (size_t i = 0; i < track_num; ++i) {
    if (!track_flags[i]) {
      unassigned_tracks->push_back(i);
    }
  }
  for (size_t i = 0; i < objects_num; ++i) {
    if (!objects_flags[i]) {
      unassigned_objects->push_back(i);
    }
  }
}

}  // namespace fusion
}  // namespace perception
}  // namespace apollo
