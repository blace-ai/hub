<h2 style="text-align:center;">Gemma (2b-v2)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [https://github.com/google/gemma_pytorch.git](https://github.com/google/gemma_pytorch.git).

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| Which is your favorite Lord of the Rings movie? | As an AI, I don't have personal preferences like a favorite movie.<br><br>However, I can tell you that the Lord of the Rings movies are generally considered to be very well-made and beloved by many fans.<br><br>Here's a breakdown of some common opinions:<br><br>* **The Fellowship of the Ring:** Many consider this the most faithful adaptation of the book, capturing the epic scope and emotional depth.<br>* **The Two Towers:** This film is praised for its action sequences and the introduction of new characters like Merry and Pippin.<br>* **The Return of the King:** This film is often considered the most visually stunning and emotionally resonant, culminating in a satisfying conclusion to the story.<br><br>Ultimately, the best Lord of the Rings movie for you depends on your personal preferences.<br><end_of_turn> |

### Demo Code
```cpp
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

```
Tested on version [v1.0.5](https://github.com/blace-ai/blace-ai/releases/tag/v1.0.5) of blace.ai sdk. Might also work on newer or older releases (check if release notes of blace.ai state breaking changes).

### Quickstart
1. Download [blace.ai SDK](https://github.com/blace-ai/blace-ai/releases/tag/v1.0.5) and unzip. In the bootstrap script `build_run_demos.ps1` (Windows) or `build_run_demos.sh` (Linux/MacOS) set the `BLACE_AI_CMAKE_DIR` environment variable to the `cmake` folder inside the unzipped SDK, e.g. `export BLACE_AI_CMAKE_DIR="<unzip_folder>/package/cmake"`. 
2. Download the model payload(s) (`.bin` files) from below and place in the same folder as the bootstrapper scripts.
3. Then run the bootstrap script with  
`powershell build_run_demo.ps1` (Windows)  
`sh build_run_demo.sh` (Linux and MacOS).  
This will build and execute the demo.

### Supported Backends
<table border="0" cellspacing="0" cellpadding="0" border-style="hidden" style="width:100%; text-align:center;">
 <thead>
    <tr>
      <th>Torchscript CPU</th>
      <th>Torchscript CUDA FP16 *</th>
      <th>Torchscript CUDA FP32 *</th>
      <th>Torchscript MPS FP16 *</th>
      <th>Torchscript MPS FP32 *</th>
      <th>ONNX CPU FP32</th>
      <th>ONNX DirectML FP32 *</th>
    </tr>
  </thead>
 <tr>
    <td>&#9989</td>
    <td>&#9989</td>
    <td>&#9989</td>
    <td>&#9989</td>
    <td>&#9989</td>
    <td>&#10060</td>
    <td>&#10060</td>
</table>
(*: Hardware Accelerated)

### Artifacts
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/a1018eaa6894d3e70685cedb136e5d9b.bin) |  | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/gemma_v2_2b_v2_v1_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/gemma_v2_2b_v2_v1_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/google/gemma_pytorch/blob/main/LICENSE)
