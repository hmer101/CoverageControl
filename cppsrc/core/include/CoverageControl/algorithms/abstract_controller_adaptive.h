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
 * \file abstract_controller_adaptive.h
 * \brief Contains the abstract class for adaptive control algorithms
 */

#ifndef CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_ABSTRACT_CONTROLLER_H_
#define CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_ABSTRACT_CONTROLLER_H_

#include <vector>

#include "CoverageControl/typedefs.h"
#include "CoverageControl/action.h"

namespace CoverageControl {

/*!
 * \addtogroup cpp_api
 * @{
 * \class AbstractController
 * @}
 * The class AbstractController is an abstract class for coverage control
 *algorithms. It provides a common interface for all coverage control
 *algorithms. Pure virtual functions: GetActions and ComputeActions
 **/
class AbstractControllerAdaptive {
 public:
  /*!
   * Pure virtual function to get the actions for the robots
   * \return The actions for the robots
   **/
  virtual std::vector<std::unique_ptr<Action>>& GetActions() = 0;

  /*!
   * Pure virtual function to compute the actions for the robots
   * \return 0 if the actions are computed successfully, 1 otherwise
   **/
  virtual int ComputeActions(int current_step) = 0;
};

} /* namespace CoverageControl */
#endif  // CPPSRC_CORE_INCLUDE_COVERAGECONTROL_ALGORITHMS_ABSTRACT_CONTROLLER_H_
