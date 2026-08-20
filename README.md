## Summary

Senior AI Framework Engineer with 6+ years at Intel, specializing in the full AI
software stack — from graph compilers (oneDNN Graph) and GPU kernel engineering
(SYCL/Triton/CUTLASS) to LLM inference and memory optimization across
CPU/GPU/HPU. Core contributor to oneDNN Graph; technical lead for large-model
memory optimization on Gaudi. Delivered up to 80% latency reduction and 93%+ MMA
utilization on production kernels.

## SKILLS

- **Languages**: C/C++, Python, SYCL, Triton, CUDA
- **Hardware Architectures**: Intel CPU, Intel Xe2/Xe3/Xe4 GPU, Intel Gaudi HPU,
  NVIDIA Hopper GPU
- **Compilers & Libraries**: torch.compile, oneDNN Graph, Triton, oneDNN
- **Frameworks**: PyTorch, DeepSpeed, Megatron-LM, vLLM, SGLang
- **GPU Kernel Engineering**: GEMM, FlashAttention, LinearAttention (GDN, KDA),
  Indexer
- **LLM Inference Optimizations**: Quantization, Kernel Fusion
- **Memory Optimization**: Graph-based memory optimization, Memory planning, GPU
  memory management, OOM analysis
- **Parallelisim**: DP, TP, EP, PP, ZeRO

## Work experience

- 2020.07 — Present — Intel APAC R&D — Senior AI Framework Engineer
  - Responsibilities: AI software framework development and model performance/memory optimization on Intel hardware platforms.
  - Summary: Core contributor to oneDNN Graph compiler, Intel CPU PaddlePaddle integration, Habana Gaudi PyTorch framework development and large-model optimization, Intel GPU operator development and large-model optimization; technical lead for Gaudi platform large-model memory optimization and partial performance optimization.

## Project experience

### 2025.07 — Present — Intel GPU operator development and model optimization\

- Description: Develop and optimize operators for large models on Intel GPU platforms; perform end-to-end performance and memory optimizations.
- Responsibilities & key achievements:
  - For Xe3 GPU architecture: continuously developed and optimized various operators in mainstream LLM models using SYCL/Triton/CUTLASS-SYCL, e.g., DeepSeek-V3.2 indexer-related operators such as mqa_logits and topk_per_row, achieving significant performance improvements (detailed metrics omitted).
  - For next-generation Xe4 GPU: validated new hardware features and developed/optimized critical operators, including Flash Attention V3/V4 prefill operators with MMA hardware utilization exceeding 93%.
  - Integrated developed operators into inference frameworks such as vLLM for end-to-end accuracy verification and performance analysis, and iteratively optimized them.

### 2023.09 — 2025.07 — Habana Gaudi HPU framework development and model optimization

- Description: Develop core PyTorch functionality on the Habana Gaudi HPU platform and optimize for large-model workloads.
- Responsibilities & key achievements:
  - Developed core modules in PyTorch such as Operator / Device / Memory / Stream / Event, and contributed to torch.compile features (e.g., identifying and fixing graph breaks, enabling CompiledAutograd).
  - Built several graph-based memory optimization methods on top of torch.compile; on models such as BERT, GPT-J, and LLaMA, memory savings up to ~15% were observed.
  - Analyzed and optimized memory usage across over 30 training/inference workloads covering models like LLaMA, DeepSeek-V3.1, GLM and frameworks including DeepSpeed, Megatron-LM, vLLM, and SGLang.
  - Developed memory analysis tools for event recording, analysis, and visualization; delivered to external customers.
  - Led optimization of the DeepSeek-R1 model (based on SGLang) for single-node 8× Gaudi3 inference using a TP8+EP8+FP8_KV_Cache scheme, achieving performance comparable to mainstream inference frameworks (detailed metrics omitted).
  - Served as the technical lead for memory optimization and was appointed the sole approver for this area in the China team.

### 2023.03 — 2023.09 — PaddlePaddle framework development and optimization on Xeon CPU (collaboration with Baidu)

- Description: Integrate and optimize PaddlePaddle on Intel Xeon CPU platforms.
- Responsibilities & key achievements:
  - Integrated oneDNN v3.1 into PaddlePaddle and used oneDNN INT8 primitives to accelerate common CNN models; e.g., ResNet-50 INT8 inference throughput reached approximately 3.2× that of FP32.

### 2020.07 — 2023.09 — oneDNN Graph compiler development

- Description: Developed the oneDNN Graph compiler based on oneDNN primitives to compile and optimize computation graphs from higher-level frameworks. The compiler uses a front/back-end decoupled design, supports multiple backends (DNNL / XeTLA) and CPU/GPU devices, and has been applied in TensorFlow/PyTorch/XLA.
- Responsibilities & key achievements:
  - Owned backend API design and implementation; led core DNNL backend architecture (DNNL IR, IR lowering, kernel fusion, layout propagation, memory planning and other optimization passes).
  - Designed and implemented large-graph compilation capability in the DNNL backend, significantly reducing overhead from small-graph scheduling and frequent memory operations; e.g., reduced ResNet50 INT8 latency by ~15% and DenseNet121 FP32 latency by ~80% on CPU.
  - Designed and implemented GPU aggressive fusion and codegen features that automatically fuse elementwise/reduction operators and generate efficient GPU executables; on Xe2-HPG GPUs, improved InstanceNorm computation graph performance to ~3.6× that of oneDNN primitive.

## Education

### Huazhong University of Science and Technology (HUST)

- 2017.09 — 2020.06 — Master of Optical Engineering (Machine Vision and Image Processing)

- Research on Low-light Image Enhancement: trained a UNet based low-light image
  enhancement model by using Generative Adversarial Networks (GANs) method.
- Model Performance Optimization and deployment: optimized performance of
  trained UNet and deployed it on NVIDIA GTX1060 GPU and developed the real-time
  camera control, image acquisition system and GUI.
- Performance Optimization on DSP: optimized performance of image processing
  operators and INT8 LeNet on TI C64X DSP

### Huazhong University of Science and Technology (HUST)

- 2013.09 — 2017.06 — Bachelor of Optoelectronic Information Science and Engineering

- National Undergraduate Electronic Design Contest — Hubei Province Third Prize (2014)
- Robomaster Robotics Competition — Huazhong regional champion (2015)

## Open-source contributions

- oneDNN contributor
- PyTorch contributor
- PaddlePaddle contributor
- SGLang contributor
- vllm-xpu-kernel contributor

## Publications

- "A Lowlight Image Enhancement Method with Ability to Suppress Noise" — Neurocomputing
- "Robot Visual Guide with Fourier-Mellin Based Visual Tracking" — Frontiers of Optoelectronics
