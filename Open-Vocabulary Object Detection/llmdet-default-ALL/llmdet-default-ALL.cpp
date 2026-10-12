#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "llmdet_v1_default_v1_ALL_export_version_v26.h"

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
  blace::ops::OpP thres = CONSTRUCT_OP(blace::ops::FromFloatOp(0.25));
  blace::ops::OpP multiple = CONSTRUCT_OP(blace::ops::FromBoolOp(true));

  // construct inference operation, returns (B,6) with 6 elems: top-left,
  // top-right, bottom-right, bottom-left, input width, input height
  auto bounding_boxes = llmdet_v1_default_v1_ALL_export_version_v26_run(
      world_tensor_orig, text_op, thres, multiple, 0, infer_args,
      util::getPathToExe().string());

  // remove width and height
  bounding_boxes = CONSTRUCT_OP(blace::ops::IndexOp(
      bounding_boxes,
      blace::ml_core::BlaceIndexVec{blace::ml_core::Slice(),
                                    blace::ml_core::Slice(0, 4)}));
  auto image_with_rectangles = CONSTRUCT_OP(blace::ops::DrawRectangles(
      world_tensor_orig, bounding_boxes, 50, 200, 50, 6));

  // write result to image file
  auto out_file = exe_path / "image_with_rectangles.png";
  image_with_rectangles =
      CONSTRUCT_OP(ops::SaveImageOp(image_with_rectangles, out_file.string()));

  // construct evaluator and evaluate
  computation_graph::GraphEvaluator evaluator(image_with_rectangles);
  auto eval_result = evaluator.evaluateToRawMemory();

  return 0;
}
