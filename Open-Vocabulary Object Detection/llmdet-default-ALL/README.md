<h2 style="text-align:center;">Llmdet (default)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [https://github.com/iSEE-Laboratory/LLMDet.git](https://github.com/iSEE-Laboratory/LLMDet.git).

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| <img src="street.jpg" alt="Image Input" width="512"/> Image Input | <img src="llmdet-default-ALL_image_with_rectangles.png" alt="Car Detections" width="512"/> Car Detections |

### Demo Code
```cpp
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
    <td>&#10060</td>
    <td>&#9989</td>
    <td>&#10060</td>
    <td>&#10060</td>
</table>
(*: Hardware Accelerated)

### Artifacts
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/000f84b63acace5c033fb7c764fcfa57.bin) |  | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/llmdet_v1_default_v1_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/llmdet_v1_default_v1_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/iSEE-Laboratory/LLMDet/blob/main/LICENSE)
