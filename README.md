## Summary

Senior AI Framework Software Engineer at Intel with 6+ years of experience in
building high-performance AI systems across CPU/GPU/HPU platforms. Working
across the full AI software stack — from LLM workload optimization (vLLM/SGLang)
and deep-learning framework development (PyTorch) to
high-performance GPU kernel engineering (SYCL/Triton/SYCL-TLA) and AI compiler
development (torch.compile, oneDNN Graph).

## SKILLS

- **LLM Inference & Serving**: vLLM, SGLang, quantization (INT8/FP8),
  parallelism (TP/EP/PP)
- **GPU Kernel Engineering**: SYCL, Triton, SYCL-TLA; GEMM, FlashAttention,
  LinearAttention; Intel/Nvidia GPU architecture; Performance modeling and
  profiling
- **AI Compiler Development**: oneDNN Graph, torch.compile; IR, operator fusion, layout
  propagation, memory planning
- **DL Framework Development**: PyTorch (core modules, torch.fx, dynamo,
  torch.compile)
- **Languages & Hardware Platforms**: C/C++, Python, SYCL, Triton, CUDA; CPU,
  GPU, HPU, Embedded DSP

## Work Experience

### Intel · Shanghai, China  2020.07 — Present

Senior AI Framework Software Engineer

AI infrastructure (kernel / compiler / framework) development and workload
optimization across Intel CPU / GPU / HPU platforms.

#### GPU Kernel Optimization & vLLM Workload Optimization — 2025.07 — Present

- Develop high-performance LLM kernels and optimize vLLM serving workloads on
  Intel Xe GPU, shipping kernels for popular LLMs across three generations of
  GPU architecture **(Xe2 / Xe3 / Xe4)** using Triton / SYCL / SYCL-TLA.
- Optimized the SYCL-TLA based **Gated DeltaNet (GDN)** kernel on Xe2 GPU,
  achieving a **geomean 1.6× speedup** over the previous hand-tuned
  implementation across a range of shapes.
- Developed key kernels for Hunyuan3-FP8 / DeepSeek-V4 models on Xe3 GPU,
  including **BF16/FP8 GQA attention prefill/decode** (**~50% of roofline**),
  the `mHC` **skinny GEMM** (TF32 + matrix engine, **~90% of roofline**), and
  `FP8/MXFP4 mqa_logits` (**~40% of roofline** in FP8, **~40% of roofline** in
  MXFP4).
- Validated new hardware features (Async DMA, Async MMA, Cluster, DSMEM) for the
  next-generation Xe4 GPU and prototyped critical kernels. The optimized
  FlashAttention V3 prefill kernel reached **93% of roofline**.
- Built a **kernel-optimization agent** on top of opencode harness, by adding
  custom command, loop control, kernel knowledge base and several
  optimization-related skills/tools; in early cases it delivered a **1.8×
  speedup** for `mqa_logits` on Xe2 and reached **~85% of roofline** for
  memory-bound operators (e.g. RMSNorm) within a single optimization round.

#### Gaudi HPU PyTorch Development & SGLang Workload Optimization — 2023.09 — 2025.07

- The [Intel Gaudi PyTorch
  Bridge](https://github.com/HabanaAI/gaudi-pytorch-bridge) enables PyTorch to
  run on the [Intel Gaudi AI Accelerator
  (HPU)](https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html)
  for large-scale LLM training and inference. Worked as a core developer on
  PyTorch development and LLM workload optimization for the Gaudi platform,
  serving as the technical lead for memory optimization.
- Developed core modules in PyTorch such as memory management, stream / event
  management, and contributed to torch.compile features, including fixing dynamo
  graph breaks, enabling CompiledAutograd, and developing optimization passes.
- Analyzed and optimized memory usage across 30+ LLM training/inference
  workloads (LLaMA, DeepSeek-V3.1, GLM) spanning different frameworks such as
  DeepSpeed, Megatron-LM, vLLM, and SGLang; built graph analysis based memory
  optimization methods on top of torch.compile, achieving up to ~15% peak memory
  reduction for various models.
- Developed memory analysis tools for device memory recording, analysis, and
  visualization; delivered to external customers.
- Optimized DeepSeek-R1 model performance on
  [SGLang](https://github.com/HabanaAI/sglang-fork) for single-node 8× Gaudi3
  inference using a TP8 + EP8 + FP8 KV-cache scheme, reaching ~90% of the
  output-token throughput of Intel's optimized vLLM on the same 8× Gaudi3 setup.

#### oneDNN Graph Compiler Development — 2020.07 — 2023.09

- [oneDNN Graph](https://uxlfoundation.github.io/oneDNN/graph_extension.html) is
  a graph compiler that extends oneDNN with a graph API to compile and optimize
  computation graphs from higher-level frameworks. It is [integrated into
  PyTorch](https://pytorch.org/blog/accelerating-inference/) to accelerate model
  inference. Contributed as a core team member, owning the DNNL backend and
  backend API.
- Designed and implemented the backend API, which was adopted as the integration
  interface for all backends including DNNL, XeTLA and other backends.
- Designed and implemented DNNL backend architecture, including DNNL IR, IR
  lowering, kernel fusion, layout propagation, memory planning and other
  optimizations.
- Designed and implemented INT8 quantization optimization in the DNNL backend,
  enabling low-precision inference across a wide range of key benchmark models
  (e.g., ResNet50, BERT, DLRM) with up to ~3× throughput over FP32 on Intel SPR
  CPU.
- Designed and implemented the large-graph compilation in the DNNL backend,
  significantly reducing overhead from small-graph scheduling and frequent
  memory operations; Reduced ResNet50 INT8 latency by ~15% and DenseNet121 FP32
  latency by ~80% on Intel SPR CPU.
- Designed and implemented the GPU aggressive fusion and codegen in the DNNL
  backend that automatically fuses elementwise/reduction operators and generates
  efficient GPU kernels; Improved InstanceNorm performance to ~3.6× that of
  oneDNN primitive on Intel ATS-M GPU.

## Education

### Huazhong University of Science and Technology (HUST)

- 2017.09 — 2020.06 — Master (Optical Engineering — Machine Vision)

- Research on Low-light Image Enhancement: trained a UNet based low-light image
  enhancement model by using Generative Adversarial Networks (GANs) method.
- Model Performance Optimization and deployment: optimized performance of
  trained UNet and deployed it on NVIDIA GTX1060 GPU and developed the real-time
  camera control, image acquisition system and GUI.
- Performance Optimization on DSP: optimized performance of image processing
  operators and INT8 LeNet on TI C64X DSP

### Huazhong University of Science and Technology (HUST)

- 2013.09 — 2017.06 — Bachelor (Optoelectronic Information Science and Engineering)

- National Undergraduate Electronic Design Contest — Hubei Province Third Prize (2014)
- Robomaster Robotics Competition — Huazhong regional champion (2015)

## Open-source contributions

- oneDNN contributor
- PyTorch contributor
- SGLang contributor
- vllm-xpu-kernel contributor

## Publications

- "A Low-light Image Enhancement Method with Ability to Suppress Noise" — Neurocomputing
- "Robot Visual Guide with Fourier-Mellin Based Visual Tracking" — Frontiers of Optoelectronics
