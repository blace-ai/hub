#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "depth_anything_v3_v2_metric_large_v1_ALL_export_version_v26.h"

using namespace blace;
int main() {
  workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "butterfly.jpg";

  auto img = CONSTRUCT_OP(ops::FromImageFileOp(photo_path.string()));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32};

  img = CONSTRUCT_OP(
      ops::Interpolate2DOp(img, 700, 1288, ml_core::AREA, false, false));

  // construct inference operation
  auto infer_op =
      depth_anything_v3_v2_metric_large_v1_ALL_export_version_v26_run(
          img, 0, infer_args, util::getPathToExe().string());

  // normalize depth to zero-one range
  auto result_depth = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(infer_op));

  auto out_file = exe_path / "depth_result.png";
  result_depth =
      CONSTRUCT_OP(ops::SaveImageOp(result_depth, out_file.string()));

  // construct evaluator and evaluate to cv::Mat
  computation_graph::GraphEvaluator evaluator(result_depth);

  auto [return_code, cv_result] = evaluator.evaluateToRawMemory();

  return 0;
}
