<h2 style="text-align:center;">Segment Anything 3 (default)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [https://github.com/facebookresearch/sam3](https://github.com/facebookresearch/sam3).

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| <img src="street.jpg" alt="Image Input" width="512"/> Image Input | <img src="sam3-default-ALL_car_matte.png" alt="Car Detections" width="512"/> Car Detections |

### Demo Code
```cpp
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
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/f3ccd60a4f2ea70ffda5cddb9bb811f6.bin) |  | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/sam3_v1_default_v6_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/sam3_v1_default_v6_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/facebookresearch/sam3?tab=License-1-ov-file#readme)
