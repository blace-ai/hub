#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "sam3_v1_default_v6_ALL_export_version_v26.h"

using namespace blace;
int main() {
  workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "street.jpg";
  auto world_tensor_orig =
      CONSTRUCT_OP(ops::FromImageFileOp(photo_path.string()));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  blace::ops::OpP text_op = CONSTRUCT_OP(blace::ops::FromTextOp("car"));
  blace::ops::OpP threshold_op = CONSTRUCT_OP(blace::ops::FromFloatOp(0.5));

  // returns all mattes with shape [c, h, w]
  auto matte = sam3_v1_default_v6_ALL_export_version_v26_run(
      world_tensor_orig, text_op, threshold_op, 0, infer_args,
      util::getPathToExe().string());

  // sum all mattes along channel dimension
  matte =
      CONSTRUCT_OP(blace::ops::SumOp(matte, std::vector<int64_t>({0}), true));

  // write result to image file
  auto out_file = exe_path / "car_matte.png";
  matte = CONSTRUCT_OP(ops::SaveImageOp(matte, out_file.string()));

  // construct evaluator and evaluate
  computation_graph::GraphEvaluator evaluator(matte);
  auto eval_result = evaluator.evaluateToRawMemory();

  return 0;
}
