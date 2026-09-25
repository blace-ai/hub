#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "ben2_v1_default_v1_ALL_export_version_v26.h"

using namespace blace;
int main() {
  ::workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "person.jpg";
  auto world_tensor_orig =
      CONSTRUCT_OP(ops::FromImageFileOp(photo_path.string()));

  // interpolate to size consumable by model
  auto interpolated = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig, 512, 896, ml_core::BICUBIC, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  // construct inference operation
  auto infer_op = ben2_v1_default_v1_ALL_export_version_v26_run(
      interpolated, 0, infer_args, util::getPathToExe().string());

  // normalize matte to zero-one range
  auto normalized_matte =
      CONSTRUCT_OP(ops::MapToRangeOp(infer_op, ml_core::ZERO_TO_ONE));

  // construct evaluator and evaluate to cv::Mat
  computation_graph::GraphEvaluator evaluator(normalized_matte);
  auto [return_code, cv_result] = evaluator.evaluateToCVMat();

  // multiply for plotting
  cv_result *= 255.;

  // save to disk and return
  auto out_file = exe_path / "matte.png";
  cv::imwrite(out_file.string(), cv_result);

  return 0;
}
