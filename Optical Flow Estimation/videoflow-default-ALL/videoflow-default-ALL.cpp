#include "blace_ai.h"
#include "videoflow_v1_default_v1_ALL_export_version_v26.h"
#include <opencv2/opencv.hpp>

using namespace blace;

int main() {
  ::workload_management::BlaceWorld blace;
  auto exe_path = util::getPathToExe();
  auto frame_0_op = CONSTRUCT_OP(
      ops::FromImageFileOp((exe_path / "videoflow_frame_0.png").string()));
  auto frame_1_op = CONSTRUCT_OP(
      ops::FromImageFileOp((exe_path / "videoflow_frame_1.png").string()));
  auto frame_2_op = CONSTRUCT_OP(
      ops::FromImageFileOp((exe_path / "videoflow_frame_2.png").string()));

  // interpolate to half size to save memory
  frame_0_op = CONSTRUCT_OP(ops::Interpolate2DOp(
      frame_0_op, 540, 960, ml_core::BILINEAR, false, true));
  frame_1_op = CONSTRUCT_OP(ops::Interpolate2DOp(
      frame_1_op, 540, 960, ml_core::BILINEAR, false, true));
  frame_2_op = CONSTRUCT_OP(ops::Interpolate2DOp(
      frame_2_op, 540, 960, ml_core::BILINEAR, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  // construct inference operation. the model returns two values, flow 1->0 and
  // flow 1->2
  auto flow_1_to_0 = videoflow_v1_default_v1_ALL_export_version_v26_run(
      frame_0_op, frame_1_op, frame_2_op, 0, infer_args,
      util::getPathToExe().string());
  auto flow_1_to_2 = videoflow_v1_default_v1_ALL_export_version_v26_run(
      frame_0_op, frame_1_op, frame_2_op, 1, infer_args,
      util::getPathToExe().string());

  // normalize optical flow to zero-one range for plotting. The model returns
  // relative offsets in -1 to 1 pixel space, so the raw values are to small to
  // plot
  flow_1_to_0 = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(flow_1_to_0));
  flow_1_to_0 = CONSTRUCT_OP(ops::ToColorOp(flow_1_to_0, ml_core::RGB));

  flow_1_to_2 = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(flow_1_to_2));
  flow_1_to_2 = CONSTRUCT_OP(ops::ToColorOp(flow_1_to_2, ml_core::RGB));

  // construct evaluator and evaluate to opencv mat
  computation_graph::GraphEvaluator evaluator_0(flow_1_to_0);
  auto [return_code_0, flow_1_to_0_cv] = evaluator_0.evaluateToCVMat();
  computation_graph::GraphEvaluator evaluator_1(flow_1_to_2);
  auto [return_code_1, flow_1_to_2_cv] = evaluator_1.evaluateToCVMat();

  // multipy for for plotting
  flow_1_to_0_cv *= 255;
  flow_1_to_2_cv *= 255;

  cv::imwrite((exe_path / "optical_flow_1_to_0.png").string(), flow_1_to_0_cv);
  cv::imwrite((exe_path / "optical_flow_1_to_2.png").string(), flow_1_to_2_cv);

  return 0;
}
