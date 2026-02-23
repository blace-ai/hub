<h2 style="text-align:center;">Coordfill (default)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [https://github.com/NiFangBaAGe/CoordFill.git](https://github.com/NiFangBaAGe/CoordFill.git).

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| <img src="example.png" alt="Image Input" width="512"/> Image Input<img src="example_mask.png" alt="Mask Input" width="512"/> Mask Input | <img src="coordfill-default-ALL_filled_image.png" alt="Output" width="512"/> Output |

### Demo Code
```cpp
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
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/52dc72b34bcc7e66e095a660a3e9296c.bin) |  | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/coordfill_v7_default_v1_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/coordfill_v7_default_v1_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/NiFangBaAGe/CoordFill/blob/master/LICENSE)
