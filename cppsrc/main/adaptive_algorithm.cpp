/*!
 * \file adaptive_algorithm.cpp
 * \brief Program to test the adaptive sampling algorithm
 *
 * This program is used to execute the adaptive sampling algorithm.
 * The environment can be initialized with default parameters. The
 * program can also take in a parameter file, a position file and an IDF file.
 * The position file contains the initial positions of the robots and the IDF
 * file contains the location of the features of interest. The program then runs
 * the adaptive sampling algorithm and outputs the final objective value.
 * ```bash
 * ./adaptive_algorithm [parameter_file] [<position_file> <idf_file>]
 * ```
 *
 */

#include <CoverageControl/algorithms/clairvoyant_adaptive.h>
#include <CoverageControl/adaptive_system.h>
#include <CoverageControl/parameters.h>
#include <CoverageControl/typedefs.h>
#include <CoverageControl/world_idf.h>
// #include <CoverageControl/algorithms/centralized_cvt.h>
// #include <CoverageControl/algorithms/decentralized_cvt.h>
// #include <CoverageControl/algorithms/near_optimal_cvt.h>
// #include <CoverageControl/algorithms/simul_explore_exploit.h>
/* #include <CoverageControl/bivariate_normal_distribution.h> */

#include <iostream>
#include <memory>
#include <string>

typedef CoverageControl::ClairvoyantAdaptive AdaptiveAlgorithm; //CoverageAlgorithm;
/* typedef CoverageControl::CentralizedCVT CoverageAlgorithm; */
/* typedef CoverageControl::DecentralizedCVT CoverageAlgorithm; */
/* typedef CoverageControl::DecentralizedCVT CoverageAlgorithm; */
/* typedef CoverageControl::NearOptimalCVT CoverageAlgorithm; */

using CoverageControl::AdaptiveSystem;
using CoverageControl::Parameters;
using CoverageControl::Point2;
using CoverageControl::PointVector;
using CoverageControl::WorldIDF;

int main(int argc, char** argv) {
  std::cout << "Started adaptive_algorithm" << std::endl;
  
  CoverageControl::CudaUtils::SetUseCuda(false);
  Parameters params;
  /* params.pSensorSize = 16; */
  if (argc >= 2) {
    std::string parameter_file = argv[1];
    params = Parameters(parameter_file);
  }

  std::unique_ptr<CoverageControl::AdaptiveSystem> env;

  if (argc == 3) {
    std::cerr << "Please provide both position and IDF files" << std::endl;
    std::cerr << "Usage: ./adaptive_algorithm [parameter_file] "
                 "[<position_file> <idf_file>]"
              << std::endl;
    return 1;
  } else if (argc == 4) {
    std::string pos_file = argv[2];
    std::string idf_file = argv[3];
    WorldIDF world_idf(params, idf_file);
    env = std::make_unique<CoverageControl::AdaptiveSystem>(params, world_idf, pos_file);
  } else {
    env = std::make_unique<CoverageControl::AdaptiveSystem>(params);
  }

  auto init_objective = env->GetObjectiveValue();
  std::cout << "Initial objective: " << init_objective << std::endl;

  AdaptiveAlgorithm algorithm(params, *env);
  auto goals = algorithm.GetGoals();

  for (int ii = 0; ii < params.pEpisodeSteps; ++ii) {
    algorithm.ComputeActions();
    auto actions = algorithm.GetActions();
    if (env->StepActions(actions)) {
      std::cout << "Invalid action" << std::endl;
      break;
    }
    if (ii % 100 == 0) {
      std::cout << "Step: " << ii << std::endl;
    }
    //env->RecordPlotData("system");
    if (algorithm.IsConverged()) {
      break;
    }
  }
  auto final_objective = env->GetObjectiveValue();
  std::cout << "Improvement %: "
            << (init_objective - final_objective) / init_objective * 100
            << std::endl;

  // std::string output_dir = "/marl_sim_basic/output_cpp";
  // std::string video_name = "system.mp4";
  // std::filesystem::create_directory(output_dir);
  // env->RenderRecordedMap(output_dir, video_name);

  // std::cout << "DONE!" << std::endl;

  return 0;
}
