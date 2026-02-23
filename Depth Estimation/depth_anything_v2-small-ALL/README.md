<h2 style="text-align:center;">Depth Anything V2 (small)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [https://github.com/DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| <img src="butterfly.jpg" alt="Depth Input" width="512"/> Depth Input | <img src="depth_anything_v2-small-ALL_depth_result.png" alt="Depth Result" width="512"/> Depth Result |

### Demo Code
```cpp
#include "blace_ai.h"
#include <opencv2/opencv.hpp>

// include the models you want to use
#include "depth_anything_v2_v8_small_v3_ALL_export_version_v26.h"

using namespace blace;
int main() {
  workload_management::BlaceWorld blace;

  // load image into op
  auto exe_path = util::getPathToExe();
  std::filesystem::path photo_path = exe_path / "butterfly.jpg";
  auto world_tensor_orig =
      CONSTRUCT_OP(ops::FromImageFileOp(photo_path.string()));

  // interpolate to size consumable by model
  auto interpolated = CONSTRUCT_OP(ops::Interpolate2DOp(
      world_tensor_orig, 700, 1288, ml_core::BICUBIC, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32};

  // construct inference operation
  auto infer_op = depth_anything_v2_v8_small_v3_ALL_export_version_v26_run(
      interpolated, 0, infer_args, util::getPathToExe().string());

  // normalize depth to zero-one range
  auto result_depth = CONSTRUCT_OP(ops::NormalizeToZeroOneOP(infer_op));

  // construct evaluator and evaluate to cv::Mat
  computation_graph::GraphEvaluator evaluator(result_depth);
  auto [return_code, cv_result] = evaluator.evaluateToCVMat();

  // multiply for plotting
  cv_result *= 255.;

  // save to disk and return
  auto out_file = exe_path / "depth_result.png";
  cv::imwrite(out_file.string(), cv_result);

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
    <td>&#9989</td>
    <td>&#9989</td>
</table>
(*: Hardware Accelerated)

### Artifacts
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/a77f7ae3aa68c06f9883308d6c116609.bin) | <td border-style="hidden" border="0"><b style="font-size:30px"><a href="https://blace-ai-public.b-cdn.net/model-payload/c54488aa61bf3e8ecfa0a63e2b297ae2.bin">ONNX Payload</a></b></td> | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/depth_anything_v2_v8_small_v3_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/depth_anything_v2_v8_small_v3_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE)
