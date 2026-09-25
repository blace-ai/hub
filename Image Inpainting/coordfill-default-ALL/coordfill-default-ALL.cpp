#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "coordfill_v7_default_v1_ALL_export_version_v26.h"

using namespace blace;

int main() {
  ::workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path image_path = exe_path / "example.png";
  auto world_tensor_orig_img =
      CONSTRUCT_OP(ops::FromImageFileOp(image_path.string()));

  std::filesystem::path mask_path = exe_path / "example_mask.png";
  auto world_tensor_orig_mask =
      CONSTRUCT_OP(ops::FromImageFileOp(mask_path.string()));
  world_tensor_orig_mask =
      CONSTRUCT_OP(ops::ToColorOp(world_tensor_orig_mask, ml_core::R));

  // interpolate to size consumable by model
  auto interpolated_img = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig_img, 640, 640, ml_core::BICUBIC, false, true));

  // interpolate to size consumable by model
  auto interpolated_mask = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig_mask, 640, 640, ml_core::BICUBIC, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  // construct inference operation
  auto infer_op = coordfill_v7_default_v1_ALL_export_version_v26_run(
      interpolated_img, interpolated_mask, 0, infer_args,
      util::getPathToExe().string());

  // write result to image file
  auto out_file = exe_path / "filled_image.png";
  infer_op = CONSTRUCT_OP(ops::SaveImageOp(infer_op, out_file.string()));

  // construct evaluator and evaluate
  computation_graph::GraphEvaluator evaluator(infer_op);
  auto eval_result = evaluator.evaluateToRawMemory();

  return 0;
}
