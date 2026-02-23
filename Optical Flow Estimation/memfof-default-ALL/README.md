<h2 style="text-align:center;">Memfof (default)</h2>

C++ (Windows, Linux, MacOS / CUDA and Metal accelerated) port of [MEMFOF](https://github.com/msu-video-group/memfof). This is a bidirectional optical flow model making better use of temporal cues.

### Example Input & Outputs
| Inputs | Outputs |
|--------|----------|
| <img src="videoflow_frame_0.png" alt="Frame 0" width="512"/> Frame 0<img src="videoflow_frame_1.png" alt="Frame 1" width="512"/> Frame 1<img src="videoflow_frame_2.png" alt="Frame 2" width="512"/> Frame 2 | <img src="memfof-default-ALL_optical_flow_1_to_0.png" alt="Optical Flow Frame 1 to 0" width="512"/> Optical Flow Frame 1 to 0<img src="memfof-default-ALL_optical_flow_1_to_2.png" alt="Optical Flow Frame 1 to 2" width="512"/> Optical Flow Frame 1 to 2 |

### Demo Code
```cpp
#include "blace_ai.h"
#include "memfof_v1_default_v3_ALL_export_version_v26.h"
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
      frame_0_op, 528, 960, ml_core::BILINEAR, false, true));
  frame_1_op = CONSTRUCT_OP(ops::Interpolate2DOp(
      frame_1_op, 528, 960, ml_core::BILINEAR, false, true));
  frame_2_op = CONSTRUCT_OP(ops::Interpolate2DOp(
      frame_2_op, 528, 960, ml_core::BILINEAR, false, true));

  // construct model inference arguments
  ml_core::InferenceArgsCollection infer_args;
  infer_args.inference_args.backends = {
      ml_core::TORCHSCRIPT_CUDA_FP16, ml_core::TORCHSCRIPT_MPS_FP16,
      ml_core::TORCHSCRIPT_CUDA_FP32, ml_core::TORCHSCRIPT_MPS_FP32,
      ml_core::ONNX_DML_FP32,         ml_core::TORCHSCRIPT_CPU_FP32};

  auto iterations = CONSTRUCT_OP(ops::FromIntOp(4));

  // construct inference operation. the model returns two values, flow 1->0 and
  // flow 1->2
  auto flow_1_to_0 = memfof_v1_default_v3_ALL_export_version_v26_run(
      frame_0_op, frame_1_op, frame_2_op, iterations, 0, infer_args,
      util::getPathToExe().string());
  auto flow_1_to_2 = memfof_v1_default_v3_ALL_export_version_v26_run(
      frame_0_op, frame_1_op, frame_2_op, iterations, 1, infer_args,
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
| [**Torchscript Payload**](https://blace-ai-public.b-cdn.net/model-payload/9aa781c87418bed7e87ccaea04d317af.bin) |  | [**Demo Project**](https://blace-ai-public.b-cdn.net/demos/memfof_v1_default_v3_ALL_export_version_v26_demo.zip) | [**Header**](https://blace-ai-public.b-cdn.net/model-defs/memfof_v1_default_v3_ALL_export_version_v26.h) |
|--------------------------------------------------------|---------------------|------------------------------------|------------------------------|
          

### [License](https://github.com/msu-video-group/memfof/blob/dev/LICENSE)
