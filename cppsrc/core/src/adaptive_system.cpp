/*
 * This file is part of the CoverageControl library
 *
 * Author: Saurav Agarwal
 * Contact: sauravag@seas.upenn.edu, agr.saurav1@gmail.com
 * Repository: https://github.com/KumarRobotics/CoverageControl
 *
 * Copyright (c) 2024, Saurav Agarwal
 *
 * The CoverageControl library is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * The CoverageControl library is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CoverageControl library. If not, see <https://www.gnu.org/licenses/>.
 */

/*!
 * \file adaptive_system.cpp
 * \brief Contains the implementation of the AdaptiveSystem class.
 */

#include <filesystem>
#include <fstream>
#include <iostream>

#include "CoverageControl/cgal/polygon_utils.h"
#include "CoverageControl/adaptive_system.h"
#include "CoverageControl/plotter.h"

namespace CoverageControl {

//using MapType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

// Add a data structure to store the sampling density in different regions of the environment
// This could be a simple 2D array or a more sophisticated data structure like a quadtree
//Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> AdaptiveSystem::sampling_density_map_;

// Helper function to calculate the gradient of the IDF map
CoverageControl::Point2 AdaptiveSystem::CalculateGradient(CoverageControl::Point2 const& position) const {
  // Get the map indices corresponding to the position
  // CoverageControl::MapUtils::MapBounds index;
  // CoverageControl::MapUtils::ComputeMapIndex(params_.pResolution, position, params_.pWorldMapSize, index);

  MapUtils::MapBounds index, offset;
  MapUtils::ComputeOffsets(params_.pResolution, position, params_.pLocalMapSize,
                           params_.pWorldMapSize, index, offset);

  // Calculate the gradient using the central difference method
  double gradient_x = 0.0;
  double gradient_y = 0.0;

  // Check if the indices are within the map bounds before calculating the gradient
  if (index.left > 0 && index.left < params_.pWorldMapSize - 1) {
    gradient_x = (GetWorldMap()(index.left + 1, index.bottom) - GetWorldMap()(index.left - 1, index.bottom)) / (2.0 * params_.pResolution);
  }

  if (index.bottom > 0 && index.bottom < params_.pWorldMapSize - 1) {
    gradient_y = (GetWorldMap()(index.left, index.bottom + 1) - GetWorldMap()(index.left, index.bottom - 1)) / (2.0 * params_.pResolution);
  }

  return CoverageControl::Point2(gradient_x, gradient_y);
}

// Helper function to calculate the gradient at a specific index in the world map
CoverageControl::Point2 AdaptiveSystem::CalculateGradientAtIndex(int x, int y) const {
  double gradient_x = 0.0;
  double gradient_y = 0.0;

  if (x > 0 && x < params_.pWorldMapSize - 1) {
    gradient_x = (GetWorldMap()(x + 1, y) - GetWorldMap()(x - 1, y)) / (2.0 * params_.pResolution);
  }

  if (y > 0 && y < params_.pWorldMapSize - 1) {
    gradient_y = (GetWorldMap()(x, y + 1) - GetWorldMap()(x, y - 1)) / (2.0 * params_.pResolution);
  }

  return CoverageControl::Point2(gradient_x, gradient_y);
}

// Function to calculate the gradient map for the entire world map
void AdaptiveSystem::CalculateGradientMap() {
  world_gradient_map_ = MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 0);

  for (int x = 1; x < params_.pWorldMapSize - 1; ++x) {
    for (int y = 1; y < params_.pWorldMapSize - 1; ++y) {
      CoverageControl::Point2 gradient = CalculateGradientAtIndex(x, y);
      world_gradient_map_(x, y) = gradient.norm();

      //std::cout << "Gradient norm at (" << x << "," << y << ") = " << gradient.norm() << std::endl;
    }
  }
}

// Modify the GetObjectiveValue() function to return a measure of uncertainty or information gain
double AdaptiveSystem::GetObjectiveValue() const {
  // Calculate the sum of the sampling density map
  //double sum_sampling_density = sampling_density_map_.sum();
  // Return the negative of the sum as the objective value (to minimize uncertainty)
   //-sum_sampling_density;

  // Get the total gradient at all of the robots' positions
  double total_gradient = 0.0;
  for (const auto& robot : robots_) {
    CoverageControl::Point2 position = robot.GetGlobalCurrentPosition();
    CoverageControl::Point2 gradient = CalculateGradient(position);
    total_gradient += gradient.norm();
  }

  return total_gradient;
}

// Add a function to simulate the robot taking a sample
void AdaptiveSystem::TakeSample(int const robot_id) {
  // Get the robot's current position
  CoverageControl::Point2 position = robots_[robot_id].GetGlobalCurrentPosition();

  // Check if the position is within the map bounds
  if (position[0] < CoverageControl::kLargeEps || position[0] > params_.pWorldMapSize - CoverageControl::kLargeEps ||
      position[1] < CoverageControl::kLargeEps || position[1] > params_.pWorldMapSize - CoverageControl::kLargeEps) {
    std::cout << "Robot " << robot_id << " is out of bounds!" << std::endl;
    return;
  }

  // Update the sampling density map
  UpdateSamplingDensity(position);

  // Simulate the robot taking a sample (e.g., update the robot's knowledge of the environment)
  // For now, this is a placeholder
  std::cout << "Robot " << robot_id << " took a sample at " << position[0] << ", " << position[1] << std::endl;

  // In a real implementation, you would update the robot's knowledge of the environment here
  // based on the IDF map and the robot's sensor characteristics.
}

// Add a function to update the sampling density based on the robot's actions
void AdaptiveSystem::UpdateSamplingDensity(CoverageControl::Point2 const& position) {
  // Get the map indices corresponding to the robot's position
  // CoverageControl::MapUtils::MapBounds index;
  // CoverageControl::MapUtils::ComputeMapIndex(params_.pResolution, position, params_.pWorldMapSize, index);

  MapUtils::MapBounds index, offset;
  MapUtils::ComputeOffsets(params_.pResolution, position, params_.pLocalMapSize,
                           params_.pWorldMapSize, index, offset);

  // Increment the sampling density in a small region around the robot's position
  int radius = 2; // Define a radius for the update
  for (int x = index.left - radius; x <= index.left + radius; ++x) {
    for (int y = index.bottom - radius; y <= index.bottom + radius; ++y) {
      // Check if the indices are within the map bounds
      if (x >= 0 && x < params_.pWorldMapSize && y >= 0 && y < params_.pWorldMapSize) {
        sampling_density_map_(x, y) += 1.0;
      }
    }
  }
}

AdaptiveSystem::AdaptiveSystem(Parameters const &params)
    : AdaptiveSystem(params, params.pNumGaussianFeatures, params.pNumPolygons,
                     params.pNumRobots) {}

AdaptiveSystem::AdaptiveSystem(Parameters const &params,
                               int const num_gaussians, int const num_robots)
    : AdaptiveSystem(params, num_gaussians, 0, num_robots) {}

AdaptiveSystem::AdaptiveSystem(Parameters const &params,
                               int const num_gaussians, int const num_polygons,
                               int const num_robots)
    : params_(params) {
  // Generate Bivariate Normal Distribution from random numbers
  std::srand(
      std::time(nullptr));  // use current time as seed for random generator
  gen_ = std::mt19937(
      rd_());  // Standard mersenne_twister_engine seeded with rd_()

  distrib_pts_ = std::uniform_real_distribution<>(
      CoverageControl::kLargeEps, params_.pWorldMapSize * params_.pResolution - CoverageControl::kLargeEps);
  std::uniform_real_distribution<> distrib_var(params_.pMinSigma,
                                               params_.pMaxSigma);
  std::uniform_real_distribution<> distrib_peak(params_.pMinPeak,
                                                params_.pMaxPeak);

  world_idf_ptr_ = std::make_shared<WorldIDF>(params_);
  for (int i = 0; i < num_gaussians; ++i) {
    CoverageControl::Point2 mean(distrib_pts_(gen_), distrib_pts_(gen_));
    double sigma = 1.0;
    if (params_.pMinSigma == params_.pMaxSigma) {
      sigma = params_.pMinSigma;
    } else {
      sigma = distrib_var(gen_);
    }
    double scale = 1.0;
    if (params_.pMinPeak == params_.pMaxPeak) {
      scale = params_.pMinPeak;
    } else {
      scale = distrib_peak(gen_);
    }
    // scale = 2.0 * M_PI * sigma * sigma * scale;
    BivariateNormalDistribution dist(mean, sigma, scale);
    world_idf_ptr_->AddNormalDistribution(dist);
  }

  std::vector<CoverageControl::PointVector> polygons;
  GenerateRandomPolygons(num_polygons, params_.pMaxVertices,
                         params_.pPolygonRadius,
                         params_.pWorldMapSize * params_.pResolution, polygons);

  for (auto &poly : polygons) {
    // double importance = distrib_peak(gen_) * 0.5;
    double importance = 0.00005;
    if (params_.pMinPeak == params_.pMaxPeak) {
      importance = params_.pMinPeak;
    } else {
      importance = distrib_peak(gen_) * importance;
    }
    PolygonFeature poly_feature(poly, importance);
    world_idf_ptr_->AddUniformDistributionPolygon(poly_feature);
  }

  world_idf_ptr_->GenerateMap();
  normalization_factor_ = world_idf_ptr_->GetNormalizationFactor();

  std::uniform_real_distribution<> env_point_dist(
      CoverageControl::kLargeEps, params_.pRobotInitDist - CoverageControl::kLargeEps);
  //robots_ = std::vector<RobotModel>(num_robots);
  robots_.reserve(num_robots);
  for (int i = 0; i < num_robots; ++i) {
    CoverageControl::Point2 start_pos(env_point_dist(gen_), env_point_dist(gen_));
    robots_.push_back(RobotModel(params_, start_pos, world_idf_ptr_));
  }
  InitSetup();
}

AdaptiveSystem::AdaptiveSystem(Parameters const &params,
                               WorldIDF const &world_idf,
                               std::string const &pos_file_name)
    : params_(params) {
  SetWorldIDF(world_idf);

  // Load initial positions
  std::ifstream file_pos(pos_file_name);
  if (!file_pos.is_open()) {
    std::cout << "Error: Could not open file " << pos_file_name << std::endl;
    exit(1);
  }
  std::vector<CoverageControl::Point2> robot_positions;
  double x, y;
  while (file_pos >> x >> y) {
    robot_positions.push_back(CoverageControl::Point2(x, y));
  }
  //robots_ = std::vector<RobotModel>(robot_positions.size());
  robots_.reserve(robot_positions.size());
  if(params_.pNumRobots != static_cast<int>(robot_positions.size())) {
    std::cerr << "Number of robots in the file does not match the number of robots in the parameters\n";
    std::cerr << "Number of robots in the file: " << robot_positions.size() << " Number of robots in the parameters: " << params_.pNumRobots << std::endl;
    exit(1);
  }
  for (size_t i = 0; i < robot_positions.size(); ++i) {
    CoverageControl::Point2 const &pos = robot_positions[i];
    robots_.push_back(RobotModel(params_, pos, world_idf_ptr_));
  }
  InitSetup();
}

AdaptiveSystem::AdaptiveSystem(Parameters const &params,
                               WorldIDF const &world_idf,
                               std::vector<CoverageControl::Point2> const &robot_positions)
    : params_(params) {
  SetWorldIDF(world_idf);

  //robots_ = std::vector<RobotModel>(robot_positions.size());
  robots_.reserve(robot_positions.size());
  for (size_t i = 0; i < robot_positions.size(); ++i) {
    CoverageControl::Point2 const &pos = robot_positions[i];
    robots_.push_back(RobotModel(params_, pos, world_idf_ptr_));
  }
  InitSetup();
}

AdaptiveSystem::AdaptiveSystem(
    Parameters const &params,
    std::vector<BivariateNormalDistribution> const &dists,
    std::vector<CoverageControl::Point2> const &robot_positions)
    : params_(params) {
  world_idf_ptr_ = std::make_shared<WorldIDF>(params_);
  world_idf_ptr_->AddNormalDistribution(dists);

  // Generate the world map
  world_idf_ptr_->GenerateMap();
  normalization_factor_ = world_idf_ptr_->GetNormalizationFactor();

  //robots_ = std::vector<RobotModel>(robot_positions.size());
  robots_.reserve(robot_positions.size());
  for (size_t i = 0; i < robot_positions.size(); ++i) {
    CoverageControl::Point2 const &pos = robot_positions[i];
    robots_.push_back(RobotModel(params_, pos, world_idf_ptr_));
  }
  InitSetup();
}

std::pair<MapType, MapType> AdaptiveSystem::GetRobotCommunicationMaps(
    size_t const id, size_t map_size) {
  std::pair<MapType, MapType> communication_maps = std::make_pair(
      MapType::Zero(map_size, map_size), MapType::Zero(map_size, map_size));
  CoverageControl::PointVector robot_neighbors_pos = GetRelativePositonsNeighbors(id);
  double center = map_size / 2. - params_.pResolution / 2.;
  CoverageControl::Point2 center_point(center, center);
  for (CoverageControl::Point2 const &relative_pos : robot_neighbors_pos) {
    CoverageControl::Point2 scaled_indices_val =
        relative_pos * map_size /
            (params_.pCommunicationRange * params_.pResolution * 2.) +
        center_point;
    int scaled_indices_x = std::round(scaled_indices_val[0]);
    int scaled_indices_y = std::round(scaled_indices_val[1]);
    CoverageControl::Point2 normalized_relative_pos = relative_pos / params_.pCommunicationRange;

    communication_maps.first(scaled_indices_x, scaled_indices_y) +=
        normalized_relative_pos[0];
    communication_maps.second(scaled_indices_x, scaled_indices_y) +=
        normalized_relative_pos[1];
  }
  return communication_maps;
}

void AdaptiveSystem::InitSetup() {
  num_robots_ = robots_.size();
  robot_positions_history_.resize(num_robots_);

  voronoi_cells_.resize(num_robots_);

  robot_global_positions_.resize(num_robots_);
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    robot_global_positions_[iRobot] =
        robots_[iRobot].GetGlobalCurrentPosition();
  }
  system_map_ =
      MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 0);
  exploration_map_ =
      MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 1);
  explored_idf_map_ =
      MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 0);

  // Initialize the gradient map
  world_gradient_map_ = MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 0);

  // Initialize the sampling density map
  sampling_density_map_ = MapType::Constant(params_.pWorldMapSize, params_.pWorldMapSize, 0);
  
  total_idf_weight_ = GetWorldMap().sum();
  relative_positions_neighbors_.resize(num_robots_);
  neighbor_ids_.resize(num_robots_);
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    relative_positions_neighbors_[iRobot].reserve(num_robots_);
    neighbor_ids_[iRobot].reserve(num_robots_);
  }
  PostStepCommands();
}

void AdaptiveSystem::PostStepCommands(size_t robot_id) {
  robot_global_positions_[robot_id] =
      robots_[robot_id].GetGlobalCurrentPosition();
  UpdateNeighbors();

  if (params_.pUpdateSystemMap) {
    MapUtils::MapBounds index, offset;
    MapUtils::ComputeOffsets(
        params_.pResolution, robot_global_positions_[robot_id],
        params_.pSensorSize, params_.pWorldMapSize, index, offset);
    explored_idf_map_.block(index.left + offset.left,
                            index.bottom + offset.bottom, offset.width,
                            offset.height) =
        GetRobotSensorView(robot_id).block(offset.left, offset.bottom,
                                           offset.width, offset.height);
    exploration_map_.block(
        index.left + offset.left, index.bottom + offset.bottom, offset.width,
        offset.height) = MapType::Zero(offset.width, offset.height);
    system_map_ = explored_idf_map_ - exploration_map_;
  }
  auto &history = robot_positions_history_[robot_id];
  if (history.size() > 0 and
      history.size() == size_t(params_.pRobotPosHistorySize)) {
    history.pop_front();
  } else {
    history.push_back(robot_global_positions_[robot_id]);
  }
}

void AdaptiveSystem::PostStepCommands() {
  UpdateRobotPositions();
  UpdateNeighbors();
  if (params_.pUpdateSystemMap) {
    UpdateSystemMap();
  }
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    auto &history = robot_positions_history_[iRobot];
    if (history.size() > 0 and
        history.size() == size_t(params_.pRobotPosHistorySize)) {
      history.pop_front();
    } else {
      history.push_back(robot_global_positions_[iRobot]);
    }
  }
}

void AdaptiveSystem::UpdateNeighbors() {
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    relative_positions_neighbors_[iRobot].clear();
    neighbor_ids_[iRobot].clear();
  }
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    for (size_t jRobot = iRobot + 1; jRobot < num_robots_; ++jRobot) {
      Point2 relative_pos =
          robot_global_positions_[jRobot] - robot_global_positions_[iRobot];
      if (relative_pos.norm() < params_.pCommunicationRange) {
        relative_positions_neighbors_[iRobot].push_back(relative_pos);
        neighbor_ids_[iRobot].push_back(jRobot);
        relative_positions_neighbors_[jRobot].push_back(-relative_pos);
        neighbor_ids_[jRobot].push_back(iRobot);
      }
    }
  }
}

bool AdaptiveSystem::StepRobotToGoal(int const robot_id, Point2 const &goal,
                                     double const speed_factor) {
  Point2 curr_pos = robots_[robot_id].GetGlobalCurrentPosition();
  Point2 diff = goal - curr_pos;
  double dist = diff.norm();
  double speed = speed_factor * dist / params_.pTimeStep;
  if (speed <= kLargeEps) {
    return 0;
  }
  speed = std::min(params_.pMaxRobotSpeed, speed);
  Point2 direction(diff);
  direction.normalize();
  if (robots_[robot_id].StepControl(direction, speed)) {
    std::cerr << "Control incorrect\n";
    return 1;
  }
  PostStepCommands();
  return 0;
}

bool AdaptiveSystem::StepRobotsToGoals(PointVector const &goals,
                                       PointVector &actions) {
  bool cont_flag = false;
  UpdateRobotPositions();
  /* #pragma omp parallel for num_threads(num_robots_) */
  for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    actions[iRobot] = Point2(0, 0);
    Point2 diff = goals[iRobot] - robot_global_positions_[iRobot];
    double dist = diff.norm();
    double speed = dist / params_.pTimeStep;
    if (speed <= kLargeEps) {
      continue;
    }
    speed = std::min(params_.pMaxRobotSpeed, speed);
    Point2 direction(diff);
    direction.normalize();
    actions[iRobot] = speed * direction;
    if (StepControl(iRobot, direction, speed)) {
      std::cerr << "Control incorrect\n";
    }
    cont_flag = true;
  }
  PostStepCommands();
  return cont_flag;
}

Point2 AdaptiveSystem::AddNoise(Point2 const pt) const {
  Point2 noisy_pt;
  noisy_pt[0] = pt[0];
  noisy_pt[1] = pt[1];
  auto noise_sigma = params_.pPositionsNoiseSigma;
  {  // Wrap noise generation in a mutex to avoid issues with random number
     // generation
     // Random number generation is not thread safe
    std::lock_guard<std::mutex> lock(mutex_);
    std::normal_distribution pos_noise{0.0, noise_sigma};
    noisy_pt += Point2(pos_noise(gen_), pos_noise(gen_));
  }

  /* std::normal_distribution pos_noise{0.0, noise_sigma}; */
  /* noisy_pt += Point2(pos_noise(gen_), pos_noise(gen_)); */
  if (noisy_pt[0] < kLargeEps) {
    noisy_pt[0] = kLargeEps;
  }
  if (noisy_pt[1] < kLargeEps) {
    noisy_pt[1] = kLargeEps;
  }
  if (noisy_pt[0] > params_.pWorldMapSize - kLargeEps) {
    noisy_pt[0] = params_.pWorldMapSize - kLargeEps;
  }
  if (noisy_pt[1] > params_.pWorldMapSize - kLargeEps) {
    noisy_pt[1] = params_.pWorldMapSize - kLargeEps;
  }
  return noisy_pt;
}
int AdaptiveSystem::WriteRobotPositions(std::string const &file_name) const {
  std::ofstream file_obj(file_name);
  if (!file_obj) {
    std::cerr << "[Error] Could not open " << file_name << " for writing."
              << std::endl;
    return 1;
  }
  file_obj << std::setprecision(kMaxPrecision);
  for (auto const &pos : robot_global_positions_) {
    file_obj << pos[0] << " " << pos[1] << std::endl;
  }
  file_obj.close();
  return 0;
}

int AdaptiveSystem::WriteRobotPositions(std::string const &file_name,
                                        PointVector const &positions) const {
  std::ofstream file_obj(file_name);
  if (!file_obj) {
    std::cerr << "[Error] Could not open " << file_name << " for writing."
              << std::endl;
    return 1;
  }
  for (auto const &pos : positions) {
    file_obj << pos[0] << " " << pos[1] << std::endl;
  }
  file_obj.close();
  return 0;
}

int AdaptiveSystem::WriteEnvironment(std::string const &pos_filename,
                                     std::string const &env_filename) const {
  WriteRobotPositions(pos_filename);
  world_idf_ptr_->WriteDistributions(env_filename);
  return 0;
}

void AdaptiveSystem::RenderRecordedMap(std::string const &dir_name,
                                       std::string const &video_name) const {
  std::string frame_dir = dir_name + "/frames/";
  std::filesystem::create_directory(frame_dir);
  Plotter plotter(frame_dir, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  // Plotter plotter_voronoi(frame_dir,
  //                         params_.pWorldMapSize * params_.pResolution,
  //                         params_.pResolution);
  // plotter_voronoi.SetScale(params_.pPlotScale);
#pragma omp parallel for
  for (size_t i = 0; i < plotter_data_.size(); ++i) {
    auto iPlotter = plotter;
    iPlotter.SetPlotName("map", i);
    /* iPlotter.PlotMap(plotter_data_[i].map, plotter_data_[i].positions,
     * plotter_data_[i].positions_history, plotter_data_[i].robot_status); */
    iPlotter.PlotMap(plotter_data_[i].map, plotter_data_[i].positions,
                     plotter_data_[i].positions_history,
                     plotter_data_[i].robot_status,
                     params_.pCommunicationRange);
    //auto iPlotterVoronoi = plotter_voronoi;
    // iPlotterVoronoi.SetPlotName("voronoi_map", i);
    // iPlotterVoronoi.PlotMap(plotter_data_[i].world_map, plotter_data_[i].positions,
    //                         plotter_data_[i].voronoi,
    //                         plotter_data_[i].positions_history);
  }
  bool ffmpeg_call =
      system(("ffmpeg -y -r 30 -i " + frame_dir +
              "map%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " +
              dir_name + "/" + video_name)
                 .c_str());
  if (ffmpeg_call) {
    std::cout << "Error: ffmpeg call failed." << std::endl;
  }
  // ffmpeg_call =
  //     system(("ffmpeg -y -r 30 -i " + frame_dir +
  //             "voronoi_map%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " +
  //             dir_name + "/voronoi_" + video_name)
  //                .c_str());
  // if (ffmpeg_call) {
  //   std::cout << "Error: ffmpeg call failed." << std::endl;
  // }
  std::filesystem::remove_all(frame_dir);
}

void AdaptiveSystem::RecordPlotData(std::vector<int> const &robot_status,
                                    std::string const &map_name) {
  PlotterData data;
  if (map_name == "world") {
    data.map = GetWorldMap();
  } else {
    data.map = system_map_;
  }
  data.positions = robot_global_positions_;
  data.positions_history = robot_positions_history_;
  data.robot_status = robot_status;
  //ComputeVoronoiCells();
  //std::vector<std::list<Point2>> voronoi;
  // auto voronoi_cells = voronoi_.GetVoronoiCells();
  // for (size_t i = 0; i < num_robots_; ++i) {
  //   std::list<Point2> cell_points;
  //   for (auto const &pos : voronoi_cells[i].cell) {
  //     cell_points.push_back(Point2(pos[0], pos[1]));
  //   }
  //   cell_points.push_back(cell_points.front());
  //   voronoi.push_back(cell_points);
  // }
  //data.voronoi = voronoi;
  data.world_map = GetWorldMap();
  plotter_data_.push_back(data);
}

void AdaptiveSystem::PlotSystemMap(std::string const &filename) const {
  std::vector<int> robot_status(num_robots_, 0);
  Plotter plotter("./", params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName(filename);
  plotter.PlotMap(system_map_, robot_global_positions_,
                  robot_positions_history_, robot_status,
                  params_.pCommunicationRange);
}

void AdaptiveSystem::PlotSystemMap(std::string const &dir_name, int const &step,
                                   std::vector<int> const &robot_status) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName("map", step);
  plotter.PlotMap(system_map_, robot_global_positions_,
                  robot_positions_history_, robot_status);
}

void AdaptiveSystem::PlotWorldMapRobots(std::string const &dir_name,
                                        std::string const &map_name) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName(map_name);
  std::vector<int> robot_status(num_robots_, 0);
  plotter.PlotMap(GetWorldMap(), robot_global_positions_,
                  robot_positions_history_, robot_status);
}

void AdaptiveSystem::PlotWorldMap(std::string const &dir_name,
                                  std::string const &map_name) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName(map_name);
  plotter.PlotMap(GetWorldMap());
}

void AdaptiveSystem::PlotInitMap(std::string const &dir_name,
                                 std::string const &map_name) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName(map_name);
  plotter.PlotMap(GetWorldMap(), robot_global_positions_);
}

void AdaptiveSystem::PlotGradientMap(std::string const &dir_name, std::string const &map_name){
  // Calculate the gradient map
  CalculateGradientMap();
  
  // Plot the gradient map
  // Get min and max gradient value
  //double min_value = world_gradient_map_.minCoeff();
  double max_value = world_gradient_map_.maxCoeff();
  //std::cout << "Gradient map min: " << min_value << " max: " << max_value << std::endl;

  // Plot the gradient map with appropriate cbrange
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName(map_name);
  plotter.PlotMap(GetWorldGradientMap(), 0.0, max_value * 1.1);
}

void AdaptiveSystem::PlotMapVoronoi(std::string const &dir_name,
                                    int const &step) {
  ComputeVoronoiCells();
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName("voronoi_map", step);
  plotter.PlotMap(GetWorldMap(), robot_global_positions_, voronoi_,
                  robot_positions_history_);
}

void AdaptiveSystem::PlotMapVoronoi(std::string const &dir_name,
                                    int const &step, Voronoi const &voronoi,
                                    PointVector const &goals) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName("map", step);
  plotter.PlotMap(GetWorldMap(), robot_global_positions_, goals, voronoi);
}

void AdaptiveSystem::PlotFrontiers(std::string const &dir_name, int const &step,
                                   PointVector const &frontiers) const {
  Plotter plotter(dir_name, params_.pWorldMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName("map", step);
  plotter.PlotMap(system_map_, robot_global_positions_,
                  robot_positions_history_, frontiers);
}

void AdaptiveSystem::PlotRobotSystemMap(std::string const &dir_name,
                                        int const &robot_id, int const &step) {
  Plotter plotter(dir_name, params_.pLocalMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetPlotName("robot_" + std::to_string(robot_id) + "_", step);
  PointVector neighbours_positions = GetRelativePositonsNeighbors(robot_id);
  for (Point2 &pos : neighbours_positions) {
    pos[0] += params_.pLocalMapSize / 2.;
    pos[1] += params_.pLocalMapSize / 2.;
  }
  plotter.PlotMap(GetRobotSystemMap(robot_id), neighbours_positions);
}

void AdaptiveSystem::PlotRobotLocalMap(std::string const &dir_name,
                                     int const &robot_id, int const &step) {
  Plotter plotter(dir_name, params_.pLocalMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetScale(params_.pPlotScale);
  plotter.SetPlotName("robot_" + std::to_string(robot_id) + "_", step);
  plotter.PlotMap(GetRobotLocalMap(robot_id));
}

void AdaptiveSystem::PlotRobotExplorationMap(std::string const &dir_name,
                                             int const &robot_id,
                                             int const &step) {
  Plotter plotter(dir_name, params_.pLocalMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetPlotName("robot_exp_" + std::to_string(robot_id) + "_", step);
  plotter.PlotMap(GetRobotExplorationMap(robot_id));
}

void AdaptiveSystem::PlotRobotSensorView(std::string const &dir_name,
                                         int const &robot_id, int const &step) {
  Plotter plotter(dir_name, params_.pSensorSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetPlotName("robot_sensor_" + std::to_string(robot_id) + "_", step);
  plotter.PlotMap(GetRobotSensorView(robot_id));
}

void AdaptiveSystem::PlotRobotObstacleMap(std::string const &dir_name,
                                          int const &robot_id,
                                          int const &step) {
  Plotter plotter(dir_name, params_.pLocalMapSize * params_.pResolution,
                  params_.pResolution);
  plotter.SetPlotName("robot_obstacle_map_" + std::to_string(robot_id) + "_",
                      step);
  plotter.PlotMap(GetRobotObstacleMap(robot_id));
}

void AdaptiveSystem::PlotRobotCommunicationMaps(std::string const &dir_name,
                                                int const &robot_id,
                                                int const &step,
                                                size_t const &map_size) {
  auto robot_communication_maps = GetRobotCommunicationMaps(robot_id, map_size);
  Plotter plotter_x(dir_name, map_size * params_.pResolution,
                    params_.pResolution);
  plotter_x.SetPlotName(
      "robot_communication_map_x_" + std::to_string(robot_id) + "_", step);
  plotter_x.PlotMap(robot_communication_maps.first);
  Plotter plotter_y(dir_name, map_size * params_.pResolution,
                    params_.pResolution);
  plotter_y.SetPlotName(
      "robot_communication_map_y_" + std::to_string(robot_id) + "_", step);
  plotter_y.PlotMap(robot_communication_maps.second);
}

PointVector AdaptiveSystem::GetRelativePositonsNeighbors(
    size_t const robot_id) {
  if (params_.pAddNoisePositions) {
    PointVector noisy_positions = GetRobotPositions();
    for (Point2 &pt : noisy_positions) {
      pt = AddNoise(pt);
    }
    PointVector relative_positions;
    for (size_t i = 0; i < num_robots_; ++i) {
      if (i == robot_id) {
        continue;
      }
      if ((noisy_positions[i] - noisy_positions[robot_id]).norm() <
          params_.pCommunicationRange) {
        relative_positions.push_back(noisy_positions[i] -
                                     noisy_positions[robot_id]);
      }
    }
    return relative_positions;
  }
  return relative_positions_neighbors_[robot_id];
}

std::vector<double> AdaptiveSystem::GetLocalVoronoiFeatures(
    int const robot_id) {
  auto const &pos = robot_global_positions_[robot_id];
  MapUtils::MapBounds index, offset;
  MapUtils::ComputeOffsets(params_.pResolution, pos, params_.pLocalMapSize,
                           params_.pWorldMapSize, index, offset);
  auto robot_map = robots_[robot_id].GetRobotMap();
  auto trimmed_local_map =
      robot_map.block(index.left + offset.left, index.bottom + offset.bottom,
                      offset.width, offset.height);
  Point2 map_size(offset.width, offset.height);

  Point2 map_translation((index.left + offset.left) * params_.pResolution,
                         (index.bottom + offset.bottom) * params_.pResolution);

  auto robot_neighbors_pos = GetRobotsInCommunication(robot_id);
  PointVector robot_positions(robot_neighbors_pos.size() + 1);

  robot_positions[0] = pos - map_translation;
  int count = 1;
  for (auto const &neighbor_pos : robot_neighbors_pos) {
    robot_positions[count] = neighbor_pos - map_translation;
    ++count;
  }
  Voronoi voronoi(robot_positions, trimmed_local_map, map_size,
                  params_.pResolution, true, 0);
  auto vcell = voronoi.GetVoronoiCell();
  return vcell.GetFeatureVector();
}
}  // namespace CoverageControl
