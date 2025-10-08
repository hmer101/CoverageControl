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

#include "CoverageControl/constants.h"
#include "CoverageControl/algorithms/abstract_controller_adaptive.h"
#include "CoverageControl/adaptive_system.h"
#include "CoverageControl/parameters.h"
#include "CoverageControl/typedefs.h"
#include "CoverageControl/action.h"

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
class ClairvoyantAdaptive : public AbstractControllerAdaptive {
 private:
  Parameters const params_;
  size_t num_robots_ = 0;
  AdaptiveSystem &env_;
  PointVector robot_global_positions_;
  PointVector goals_;
  std::vector<std::unique_ptr<Action>> actions_;
  PointVector visited_goals_;  //!< List of previously visited goals (shared across all robots)

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

  std::vector<std::unique_ptr<Action>>& GetActions() { return actions_; }

  auto GetGoals() { return goals_; }

  Point2 ComputeGoal(size_t iRobot, PointVector const& active_goals) {
    // Implement the logic to find the largest importance in a neighborhood around the robot
    MapType const &world_map = env_.GetWorldMap(); // GetWorldMapMutable

    Point2 current_position = env_.GetRobotPosition(iRobot);
    Point2 best_goal = current_position;
    double max_importance = -1.0;

    // Define a minimum distance threshold to consider a location as "visited"
    double visited_threshold = params_.pResolution * params_.pSampleRadius;

    // Calculate the search bounds to stay within the map
    double world_size = params_.pWorldMapSize * params_.pResolution;
    double max_search_dist = params_.pMaxSearchRadius;

    // Clamp search radius to map boundaries
    int x_min = static_cast<int>(std::max(-max_search_dist, -current_position[0]) / params_.pResolution);
    int x_max = static_cast<int>(std::min(max_search_dist, world_size - current_position[0]) / params_.pResolution);
    int y_min = static_cast<int>(std::max(-max_search_dist, -current_position[1]) / params_.pResolution);
    int y_max = static_cast<int>(std::min(max_search_dist, world_size - current_position[1]) / params_.pResolution);

    // Search for the largest importance in a neighborhood around the robot
    for (int x = x_min; x <= x_max; ++x) {
      for (int y = y_min; y <= y_max; ++y) {
        Point2 candidate_position;
        candidate_position[0] = current_position[0] + x * params_.pResolution;
        candidate_position[1] = current_position[1] + y * params_.pResolution;
        // Position is guaranteed to be within bounds due to clamping above

        // Check if this location has been visited before
        bool already_visited = false;
        for (auto const &visited_goal : visited_goals_) {
          if ((candidate_position - visited_goal).norm() < visited_threshold) {
            already_visited = true;
            break;
          }
        }
        if (already_visited) {
          continue;  // Skip this candidate
        }

        // Check if this location is an active goal for another robot
        bool is_active_goal = false;
        for (auto const &goal : active_goals) {
            if ((candidate_position - goal).norm() < visited_threshold) {
                is_active_goal = true;
                break;
            }
        }
        if (is_active_goal) {
            continue; // Skip this candidate
        }

        // Find the IDF value at the candidate position
        // Convert to indices
        int i = static_cast<int>(candidate_position[0] / params_.pResolution);
        int j = static_cast<int>(candidate_position[1] / params_.pResolution);

        float importance = world_map(i, j);

        // Update the best goal if the current importance is larger than the importance found so far
        if (importance > max_importance) {
          max_importance = importance;
          best_goal = candidate_position;
        }
      }
    }

    // Update the goal for the robot
    goals_[iRobot] = best_goal;
    return best_goal;
  }

  void ComputeGoals() {
    // Implement the logic to find the largest importance in a neighborhood around the robot
    //MapType const &world_map = env_.GetWorldMap(); // GetWorldMapMutable

    // For each robot, compute the best goal
    //#pragma omp parallel for
    PointVector active_goals;

    for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
      Point2 newGoal = ComputeGoal(iRobot, active_goals); //Point2 best_goal = 
      active_goals.push_back(newGoal);
      //goals_[iRobot] = best_goal;
    }
  }


  int ComputeActions(int current_step) {
    is_converged_ = false; //true;  // Assume convergence until a robot takes an action
    robot_global_positions_ = env_.GetRobotPositions();

    // Build a list of all goals that are currently being pursued
    PointVector active_goals;
    for (size_t i = 0; i < num_robots_; ++i) {
        if (actions_[i] != nullptr) {
            active_goals.push_back(actions_[i]->GetTargetPosition());
        }
    }

    for (size_t iRobot = 0; iRobot < num_robots_; ++iRobot) {
      Point2 current_pos = robot_global_positions_[iRobot];
      auto& current_action = actions_[iRobot];

      // State 1: Robot is MOVING
      if (current_action != nullptr && current_action->GetActionType() == "Move") {
        if (current_action->IsComplete(current_step, params_, current_pos)) {
          // Reached goal, transition to sampling state //current_action = nullptr;  
          current_action = std::make_unique<SampleAction>(current_pos);
          current_action->SetStartTime(current_step);
          is_converged_ = false;  // Taking an action
        } else {
          is_converged_ = false;  // Still moving
          continue;               // Continue with the current MoveAction
        }
      }

      // State 2: Robot is SAMPLING
      else if (current_action != nullptr && current_action->GetActionType() == "Sample") {
        auto* sample_action = static_cast<SampleAction*>(current_action.get());
        if (sample_action->IsComplete(current_step, params_, current_pos)) {
          visited_goals_.push_back(goals_[iRobot]); // GOAL IS VISITED ONLY AFTER SAMPLING IS COMPLETE
          
          // Sampling done, transition to moving state
          //current_action = nullptr;  
          ComputeGoal(iRobot, active_goals);
          Point2 new_goal = goals_[iRobot];

          current_action = std::make_unique<MoveAction>(new_goal);
          active_goals.push_back(new_goal); // Add to active goals for this step
          is_converged_ = false;  // Taking an action
        } else {
          // Start the sampling timer if it hasn't been started
          if (!sample_action->IsStarted()) {
            sample_action->SetStartTime(current_step);
          }
          is_converged_ = false;  // Still sampling
          continue;               // Continue with the current SampleAction
        }
      }

      // State 3: Robot is IDLE (no action) -> Decide what to do next
      else if (current_action == nullptr) {
        // The robot is idle, so it needs a new goal.
        ComputeGoal(iRobot, active_goals);
        Point2 new_goal = goals_[iRobot];

        current_action = std::make_unique<MoveAction>(new_goal);
        active_goals.push_back(new_goal); // Add to active goals for this step
        is_converged_ = false;  // Taking an action
      }
    }
    return 0;
  }

  bool IsConverged() const { return is_converged_; }
};

}  // namespace CoverageControl
#endif  // CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_CLAIRVOYANT_ADAPTIVE_H_
