#include "blace_ai.h"
#include <fstream>
#include <iostream>

// include the models you want to use
#include "gemma_v2_2b_v2_v1_ALL_export_version_v26.h"

using namespace blace;
int main() {
  ::workload_management::BlaceWorld blace;

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  std::vector<std::string> questions = {
      "What is the answer to life?", "Will ai rule the world?",
      "Which is your favorite lord of the rings movie?"};

  for (auto str : questions) {
    auto text_t = CONSTRUCT_OP(ops::FromTextOp(str));

    auto output_len = CONSTRUCT_OP(ops::FromIntOp(200));
    auto temperature = CONSTRUCT_OP(ops::FromFloatOp(0.));
    auto top_p = CONSTRUCT_OP(ops::FromFloatOp(0.9));
    auto top_k = CONSTRUCT_OP(ops::FromIntOp(50));

    // construct inference operation
    auto infer_op = gemma_v2_2b_v2_v1_ALL_export_version_v26_run(
        text_t, output_len, temperature, top_p, top_k, 0, infer_args,
        util::getPathToExe().string());

    computation_graph::GraphEvaluator evaluator(infer_op);
    auto [return_code, answer] = evaluator.evaluateToString();
    std::cout << "Answer: " << answer << std::endl;

    // writes text to file
    std::ofstream out("answer.txt");
    out << answer;
    out.close();
  }

  return 0;
}
