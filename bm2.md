## Benchmark

### Backends
CPU: ncnn, ONNXRuntime, OpenVINO

GPU: TensorRT, PPLNN

### Latency benchmark
#### Platform
- Ubuntu 18.04
- Cuda 11.3
- TensorRT 7.2.3.4
- Docker 20.10.8
- NVIDIA tesla T4 tensor core GPU for TensorRT.

#### Other settings
- Static graph
- Batch size 1
- Synchronize devices after each inference.
- We count the average inference performance of 100 images of the dataset.
- Warm up. For classification, we warm up 1010 iters. For other codebases, we warm up 10 iters.
- Input resolution varies for different datasets of different codebases. All inputs are real images except for `mmediting` because the dataset is not large enough.


Users can directly test the speed through [how_to_measure_performance_of_models.md](tutorials/how_to_measure_performance_of_models.md). And here is the benchmark in our environment.

<svg xmlns="http://www.w3.org/2000/svg">
  <foreignObject width="100%" height="100%">
    <link rel="stylesheet" href="./style.css">
    <table class="docutils">
      <thead>
        <tr>
          <th align="center" colspan="3">MMSeg</th>
          <th align="center">Pytorch</th>
          <th align="center">ONNXRuntime</th>
          <th align="center" colspan="3">TensorRT</th>
          <th align="center">PPLNN</th>
          <th align="center"></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td align="center">Model</td>
          <td align="center">Dataset</td>
          <td align="center">Metrics</td>
          <td align="center">fp32</td>
          <td align="center">fp32</td>
          <td align="center">fp32</td>
          <td align="center">fp16</td>
          <td align="center">int8</td>
          <td align="center">fp16</td>
          <td>model config file</td>
        </tr>
        <tr>
          <td align="center">FCN</td>
          <td align="center">Cityscapes</td>
          <td align="center">mIoU</td>
          <td align="center">72.25</td>
          <td align="center">-</td>
          <td align="center">72.36</td>
          <td align="center">72.35</td>
          <td align="center">74.19</td>
          <td align="center">-</td>
          <td>$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
        </tr>
        <tr>
          <td align="center">PSPNet</td>
          <td align="center">Cityscapes</td>
          <td align="center">mIoU</td>
          <td align="center">78.55</td>
          <td align="center">-</td>
          <td align="center">78.26</td>
          <td align="center">78.24</td>
          <td align="center">77.97</td>
          <td align="center">-</td>
          <td>$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
        </tr>
        <tr>
          <td align="center">deeplabv3</td>
          <td align="center">Cityscapes</td>
          <td align="center">mIoU</td>
          <td align="center">79.09</td>
          <td align="center">-</td>
          <td align="center">79.12</td>
          <td align="center">79.12</td>
          <td align="center">78.96</td>
          <td align="center">-</td>
          <td>$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py</td>
        </tr>
        <tr>
          <td align="center">deeplabv3+</td>
          <td align="center">Cityscapes</td>
          <td align="center">mIoU</td>
          <td align="center">79.61</td>
          <td align="center">-</td>
          <td align="center">79.6</td>
          <td align="center">79.6</td>
          <td align="center">79.43</td>
          <td align="center">-</td>
          <td>$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py</td>
        </tr>
        <tr>
          <td align="center">Fast-SCNN</td>
          <td align="center">Cityscapes</td>
          <td align="center">mIoU</td>
          <td align="center">70.96</td>
          <td align="center">-</td>
          <td align="center">70.93</td>
          <td align="center">70.92</td>
          <td align="center">66.0</td>
          <td align="center">-</td>
          <td>$MMSEG_DIR/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py</td>
        </tr>
      </tbody>
    </table>
  </foreignObject>
</svg>

### Notes
- As some datasets contain images with various resolutions in codebase like MMDet. The speed benchmark is gained through static configs in MMDeploy, while the performance benchmark is gained through dynamic ones.

- Some int8 performance benchmarks of TensorRT require Nvidia cards with tensor core, or the performance would drop heavily.

- DBNet uses the interpolate mode `nearest` in the neck of the model, which TensorRT-7 applies a quite different strategy from Pytorch. To make the repository compatible with TensorRT-7, we rewrite the neck to use the interpolate mode `bilinear` which improves final detection performance. To get the matched performance with Pytorch, TensorRT-8+ is recommended, which the interpolate methods are all the same as Pytorch.
