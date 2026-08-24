## Summary

Senior AI Framework Software Engineer at Intel with 6+ years of experience in
building high-performance AI systems across CPU/GPU/HPU platforms. Working
across the full AI software stack — from LLM workload optimization (vLLM/SGLang)
and deep-learning framework development (PyTorch/PaddlePaddle) to
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
  torch.compile), PaddlePaddle
- **Languages & Hardware Platforms**: C/C++, Python, SYCL, Triton, CUDA; CPU,
  GPU, HPU, Embedded DSP

## Work Experience

### Intel · Shanghai, China  2020.07 — Present

Senior AI Framework Software Engineer

AI infrastructure (kernel / compiler / framework) development and workload
optimization across Intel CPU / GPU / HPU platforms.

#### GPU Kernel Optimization & vLLM Workload Optimization — 2025.07 — Present

- Develop and optimize operators for large models on Intel GPU platforms; perform end-to-end performance and memory optimizations.
- For Xe3 GPU architecture: continuously developed and optimized various operators in mainstream LLM models using SYCL/Triton/SYCL-TLA, e.g., DeepSeek-V3.2 indexer-related operators such as mqa_logits and topk_per_row, achieving significant performance improvements (detailed metrics omitted).
- For next-generation Xe4 GPU: validated new hardware features and developed/optimized critical operators, including Flash Attention V3/V4 prefill operators with MMA hardware utilization exceeding 93%.
- Integrated developed operators into inference frameworks such as vLLM for end-to-end accuracy verification and performance analysis, and iteratively optimized them.

#### Gaudi HPU PyTorch Development & SGLang Workload Optimization — 2023.09 — 2025.07

- Develop core PyTorch functionality on the Habana Gaudi HPU platform and optimize for large-model workloads.
- Developed core modules in PyTorch such as Operator / Device / Memory / Stream / Event, and contributed to torch.compile features (e.g., identifying and fixing graph breaks, enabling CompiledAutograd).
- Built several graph-based memory optimization methods on top of torch.compile; on models such as BERT, GPT-J, and LLaMA, memory savings up to ~15% were observed.
- Analyzed and optimized memory usage across over 30 training/inference workloads covering models like LLaMA, DeepSeek-V3.1, GLM and frameworks including DeepSpeed, Megatron-LM, vLLM, and SGLang.
- Developed memory analysis tools for event recording, analysis, and visualization; delivered to external customers.
- Led optimization of the DeepSeek-R1 model (based on SGLang) for single-node 8× Gaudi3 inference using a TP8+EP8+FP8_KV_Cache scheme, achieving performance comparable to mainstream inference frameworks (detailed metrics omitted).
- Served as the technical lead for memory optimization and was appointed the sole approver for this area in the China team.

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
- PaddlePaddle contributor
- SGLang contributor
- vllm-xpu-kernel contributor

## Publications

- "A Low-light Image Enhancement Method with Ability to Suppress Noise" — Neurocomputing
- "Robot Visual Guide with Fourier-Mellin Based Visual Tracking" — Frontiers of Optoelectronics
