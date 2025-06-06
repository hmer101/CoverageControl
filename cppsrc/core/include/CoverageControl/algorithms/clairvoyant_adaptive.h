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
 * \file clairvoyant_adaptive.h
 * \brief Clairvoyant adaptive sampling algorithm
 */

#ifndef CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_CLAIRVOYANT_ADAPTIVE_H_
#define CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_CLAIRVOYANT_ADAPTIVE_H_

#include <omp.h>

#include <algorithm>
#include <vector>

#include "CoverageControl/algorithms/abstract_controller.h"
#include "CoverageControl/adaptive_system.h"
#include "CoverageControl/parameters.h"
#include "CoverageControl/typedefs.h"

namespace CoverageControl {

/*!
 * \addtogroup cpp_api
 * @{
 * \class ClairvoyantAdaptive
 * @}
 * Clairvoyant adaptive sampling algorithm
 * The algorithm has knowledge of the entire map in a centralized manner.
 * It selects the next sampling location based on the largest gradient at the
 * edge of what the robot knows.
 */
class ClairvoyantAdaptive : public AbstractController {
 private:
  Parameters const params_;
  size_t num_robots_ = 0;
  AdaptiveSystem &env_;
  PointVector robot_global_positions_;
  PointVector goals_, actions_;

  bool is_converged_ = false;

 public:
  ClairvoyantAdaptive(Parameters const &params, AdaptiveSystem &env)
      : ClairvoyantAdaptive(params, params.pNumRobots, env) {} 
  ClairvoyantAdaptive(Parameters const &params, size_t const &num_robots,
                 AdaptiveSystem &env)
      : params_{params}, num_robots_{num_robots}, env_{env} {
    robot_global_positions_ = env_.GetRobotPositions();
    actions_.resize(num_robots_);
    goals_ = robot_global_positions_;
    ComputeGoals();
  }

  PointVector GetActions() { return actions_; }

  auto GetGoals() { return goals_; }

  void ComputeGoals() {
    // Implement the logic to find the largest gradient at the edge of what the robot knows
    // for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
    //   Point2 current_position = env_.GetRobotPosition(iRobot);
    //   Point2 best_goal = current_position;
    //   double max_gradient = -1.0; 
      
    //   // Search for the largest gradient in a neighborhood around the robot
    //   int search_radius = params_.pResolution*50; // Define a radius for the search
    //   for (int x = -search_radius; x <= search_radius; ++x) {
    //     for (int y = -search_radius; y <= search_radius; ++y) {
    //       Point2 candidate_position;
    //       candidate_position[0] = current_position[0] + x * params_.pResolution;
    //       candidate_position[1] = current_position[1] + y * params_.pResolution;

    //       // Check if the candidate position is within the map bounds
    //       if (candidate_position[0] >= 0 && candidate_position[0] < params_.pWorldMapSize * params_.pResolution &&
    //           candidate_position[1] >= 0 && candidate_position[1] < params_.pWorldMapSize * params_.pResolution) {

    //         // Calculate the gradient at the candidate position
    //         Point2 gradient = env_.CalculateGradient(candidate_position);
    //         double gradient_magnitude = gradient.norm();

    //         // Update the best goal if the current gradient is larger than the maximum gradient found so far
    //         if (gradient_magnitude > max_gradient) {
    //           max_gradient = gradient_magnitude;
    //           best_goal = candidate_position;
    //         }
    //       }
    //     }
    //   }
    //   goals_[iRobot] = best_goal;
    // }

    // Implement the logic to find the largest importance in a neighborhood around the robot
    MapType const &world_map = env_.GetWorldMap(); // GetWorldMapMutable

    for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
      Point2 current_position = env_.GetRobotPosition(iRobot);
      Point2 best_goal = current_position;
      double max_importance = -1.0; 
      
      // Search for the largest gradient in a neighborhood around the robot
      int search_radius = params_.pResolution*50; // Define a radius for the search
      for (int x = -search_radius; x <= search_radius; ++x) {
        for (int y = -search_radius; y <= search_radius; ++y) {
          Point2 candidate_position;
          candidate_position[0] = current_position[0] + x * params_.pResolution;
          candidate_position[1] = current_position[1] + y * params_.pResolution;

          // Check if the candidate position is within the map bounds
          if (candidate_position[0] >= 0 && candidate_position[0] < params_.pWorldMapSize * params_.pResolution &&
              candidate_position[1] >= 0 && candidate_position[1] < params_.pWorldMapSize * params_.pResolution) {

            // Find the IDF value at the candidate position
            // Convert to indices
            int i = static_cast<int>(candidate_position[0] / params_.pResolution);
            int j = static_cast<int>(candidate_position[1] / params_.pResolution);
            
             // Safety check
            //if (i >= 0 && i < params_.pWorldMapSize && j >= 0 && j < params_.pWorldMapSize) {
            float importance = world_map(i, j);

            // Update the best goal if the current importance is larger than the importance found so far
            if (importance > max_importance) {
              max_importance = importance;
              best_goal = candidate_position;
            }
            //}
          }
        }
      }
      goals_[iRobot] = best_goal;
    }
  }

  int ComputeActions() {
    is_converged_ = true;
    robot_global_positions_ = env_.GetRobotPositions();
    ComputeGoals();
    for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
      actions_[iRobot] = Point2(0, 0);
      Point2 diff = goals_[iRobot] - robot_global_positions_[iRobot];
      double dist = diff.norm();
      if (dist < kEps) {
        continue;
      }
      if (env_.CheckOscillation(iRobot)) {
        continue;
      }
      double speed = dist / params_.pTimeStep;
      speed = std::min(params_.pMaxRobotSpeed, speed);
      Point2 direction(diff);
      direction.normalize();
      actions_[iRobot] = speed * direction;
      is_converged_ = false;

    }
    return 0;
  }

  bool IsConverged() const { return is_converged_; }
};

}  // namespace CoverageControl
#endif  // CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_CLAIRVOYANT_ADAPTIVE_H_
