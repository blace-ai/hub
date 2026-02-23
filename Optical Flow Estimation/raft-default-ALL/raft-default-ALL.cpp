#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "raft_v1_default_v2_ALL_export_version_v26.h"

using namespace blace;

std::shared_ptr<RawMemoryObject> memory_from_file(std::string file) {
  // read image into memory
  cv::Mat image = cv::imread(file, cv::IMREAD_COLOR);

  // construct a hash from the filename
  ml_core::BlaceHash random_hash(file);

  // construct the memory object. We set copy_memory to true, since image will
  // be out-of-scope upon method return and therefore we need to take ownership
  // of the data
  RawMemoryObject raw_mem((void *)image.data, ml_core::DataTypeEnum::BLACE_BYTE,
                          ml_core::ColorFormatEnum::BGR,
                          std::vector<int64_t>{1, image.rows, image.cols, 3},
                          ml_core::BHWC, ml_core::ZERO_TO_255, ml_core::CPU,
                          random_hash, true);

  return std::make_shared<RawMemoryObject>(raw_mem);
}

int main() {
  ::workload_management::BlaceWorld blace;
  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path frame_0 = exe_path / "raft_frame_0.png";
  std::filesystem::path frame_1 = exe_path / "raft_frame_1.png";

  auto frame_0_mem = memory_from_file(frame_0.string());
  auto frame_1_mem = memory_from_file(frame_1.string());

  auto frame_0_op = CONSTRUCT_OP(ops::FromRawMemoryOp(frame_0_mem));
  auto frame_1_op = CONSTRUCT_OP(ops::FromRawMemoryOp(frame_1_mem));

  auto num_flow_updates = CONSTRUCT_OP(ops::FromIntOp(12));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  // construct inference operation
  auto infer_op = raft_v1_default_v2_ALL_export_version_v26_run(
      frame_0_op, frame_1_op, num_flow_updates, 0, infer_args,
      util::getPathToExe().string());

  // normalize optical flow to zero-one range for plotting. The model returns
  // relative offsets in -1 to 1 pixel space, so the raw values are to small to
  // plot
  infer_op = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(infer_op));

  // convert uv color to rgb (by U->R, V->G, 1->B)
  infer_op = CONSTRUCT_OP(ops::ToColorOp(infer_op, ml_core::RGB));

  // we prepare the result for later copy to cv::Mat. The values set here are
  // based on implicit knowledge of cv::Mat internal data storage.
  auto normalized_matte = CONSTRUCT_OP(ops::PrepareForHostCopyOP(
      infer_op, ml_core::BLACE_BYTE, ml_core::RGB, ml_core::HWC,
      ml_core::ZERO_TO_255, ml_core::CPU));

  // construct evaluator and evaluate to raw memory object
  computation_graph::GraphEvaluator evaluator(normalized_matte);
  auto [return_code, raw_mem] = evaluator.evaluateToRawMemory();

  // get the sizes
  int w = raw_mem->get_memory_sizes()[1];
  int h = raw_mem->get_memory_sizes()[0];

  // initialize an empty cv::Mat
  cv::Mat cv_mat(h, w, CV_8UC3);
  cv_mat.setTo(cv::Scalar(0, 0, 0));

  // and copy the memory
  std::memcpy(cv_mat.data, raw_mem->get_data_ptr(), raw_mem->get_memory_size());

  // save to disk and return
  auto out_file = exe_path / "optical_flow.png";
  cv::imwrite(out_file.string(), cv_mat);

  return 0;
}
