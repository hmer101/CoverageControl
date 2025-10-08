/*
 * This file is part of the AdaptiveSampling library
 *
 * Author: Harvey Merton
 * Contact: hmer101@mit.edu
 *
 * Copyright (c) 2025, Harvey Merton
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*!
 * \file action.h
 * \brief Contains the Action base class and its subclasses (Move, Sample)
 */

#ifndef CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ACTION_H_
#define CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ACTION_H_

#include <memory>
#include "CoverageControl/typedefs.h"
#include "CoverageControl/parameters.h"

namespace CoverageControl {

/*!
 * \addtogroup cpp_api
 * @{
 * \class Action
 * @}
 * \brief Base class for robot actions
 *
 * This abstract base class represents different types of actions that a robot
 * can perform, such as moving to a location or sampling at a location.
 */
class Action {
 protected:
  Point2 target_position_;  //!< Target position for the action
  int t_start_;            //!< Step when action was started (-1 if not started)
  int robot_id_;           //!< ID of the robot associated with this action

 public:
  Action(Point2 const &target_position)
      : target_position_(target_position), t_start_(-1), robot_id_(-1) {}

  Action(Point2 const &target_position, int robot_id)
      : target_position_(target_position), t_start_(-1), robot_id_(robot_id) {}

  virtual ~Action() = default;

  //! Get the robot ID associated with this action
  int GetRobotId() const { return robot_id_; }

  //! Set the robot ID associated with this action
  void SetRobotId(int robot_id) { robot_id_ = robot_id; }

  //! Get the target position for this action
  Point2 GetTargetPosition() const { return target_position_; }

  //! Set the target position for this action
  void SetTargetPosition(Point2 const &position) { target_position_ = position; }

  //! Get the start time step of this action
  int GetStartTime() const { return t_start_; }

  //! Set the start time step of this action
  void SetStartTime(int t_start) { t_start_ = t_start; }

  //! Check if action has been started
  bool IsStarted() const { return t_start_ >= 0; }

  //! Pure virtual function to check if action is complete
  //! \param current_step Current simulation step
  //! \param params Parameters object for accessing duration settings
  //! \param current_position Current robot position
  //! \return true if action is complete, false otherwise
  virtual bool IsComplete(int current_step, Parameters const &params,
                         Point2 const &current_position) const = 0;

  //! Pure virtual function to get the velocity action for this step
  //! \param current_position Current robot position
  //! \param params Parameters object
  //! \return Velocity vector for this step
  virtual Point2 GetVelocityAction(Point2 const &current_position,
                                  Parameters const &params) const = 0;

  //! Get the type of action as a string (for debugging)
  virtual std::string GetActionType() const = 0;
};

/*!
 * \addtogroup cpp_api
 * @{
 * \class MoveAction
 * @}
 * \brief Action class for moving to a target location
 *
 * This action moves the robot towards a target position without sampling.
 * The action completes when the robot reaches the target position.
 */
class MoveAction : public Action {
 public:
  explicit MoveAction(Point2 const &target_position)
      : Action(target_position) {}

  bool IsComplete(int current_step, Parameters const &params,
                 Point2 const &current_position) const override {
    Point2 diff = target_position_ - current_position;
    return diff.norm() < kEps;  // Action complete when robot reaches target
  }

  Point2 GetVelocityAction(Point2 const &current_position,
                          Parameters const &params) const override {
    Point2 diff = target_position_ - current_position;
    double dist = diff.norm();

    if (dist < kEps) {
      return Point2(0, 0);  // Already at target
    }

    double speed = dist / params.pTimeStep;
    speed = std::min(params.pMaxRobotSpeed, speed);
    Point2 direction = diff.normalized();
    return speed * direction;
  }

  std::string GetActionType() const override {
    return "Move";
  }
};

/*!
 * \addtogroup cpp_api
 * @{
 * \class SampleAction
 * @}
 * \brief Action class for sampling at a location
 *
 * This action keeps the robot stationary at its current position while sampling
 * for a specified duration (params.pSampleDuration steps). The robot should already
 * be at the target location when this action is created (use MoveAction first).
 */
class SampleAction : public Action {
 private:
  mutable bool sample_taken_;  //!< Flag to track if sample has been taken

 public:
  explicit SampleAction(Point2 const &target_position)
      : Action(target_position), sample_taken_(false) {}

  // SAMPLING SPECIFIC METHODS
  void TakeSample() const{ // TODO: make this non-const?
    sample_taken_ = true;
  }
  
  bool ReadyToSample(int current_step, Parameters const &params) const{
    // If not started yet or sample already taken, we're not ready to sample
    if (!IsStarted() || sample_taken_) {
      return false;
    }

    // Check if sampling duration has elapsed
    int elapsed_time = (current_step - t_start_)* params.pTimeStep;
    if (elapsed_time >= params.pSampleDuration) {
      return true;
    }
    return false;
  }

  // GENERIC ACTION METHODS
  bool IsComplete(int current_step, Parameters const &params,
                 Point2 const &current_position) const override {
    // If sample has been taken, action is complete
    if (sample_taken_) {
      return true;
    }else{
      return false;
    }
  }


  Point2 GetVelocityAction(Point2 const &current_position,
                          Parameters const &params) const override {
    // Robot should stay stationary at current position while sampling
    return Point2(0, 0);
  }

  std::string GetActionType() const override {
    return "Sample";
  }
};

} // namespace CoverageControl

#endif  // CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ACTION_H_