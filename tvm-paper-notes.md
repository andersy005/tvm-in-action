# [Summary of TVM: End-to-End Optimization Stack for Deep Learning](https://arxiv.org/abs/1802.04799)

## Abstract

- Scalable frameworks, such as TensorFlow, MXNet, Caffe, and PyTorch are optimized for a narrow range of serve-class GPUs.
- Deploying workloads to other platforms such as mobile phones, IoT, and specialized accelarators(FPGAs, ASICs) requires laborious manual effort.
- TVM is an end-to-end optimization stack that exposes:
  - graph-level
  - operator-level optimizations
  ---> to provide performance portability to deep learning workloads across diverse hardware back-ends.

## Introduction

- The number and diversity of specialized deep learning (DL) accelerators pose an adoption challenge
  - They introduce new hardware abstractions that modern compilers and frameworks are ill-equipped to deal with.

- Providing support in various DL frameworks for diverse hardware back-ends in the present ad-hoc fashion is **unsustainable**.

- Hardware targets significantly diverge in terms of memory organization, compute, etc..

![](https://i.imgur.com/XRSZMt0.png)

- *The Goal*: **easily deploy DL workloads to all kinds of hardware targets, including embedded devives, GPUs, FPGAs, ASCIs (e.g, the TPU).**

- Current DL frameworks rely on a **computational graph intermediate representation** to implement optimizations such as:
  - auto differentiation
  - dynamic memory management

- **Graph-level optimizations** are often too high-level to handle hardware back-end-specific **operator transformations**.
- **Current operator-level libraries** that DL frameworks rely on are:
  - too rigid
  - specialized
  ---> to be easily ported **across hardware devices**

- To address these weaknesses, we need a **compiler framework** that can expose optimization opportunities across both
  - graph-level and
  - operator-level
  ---> to deliver competitive performance across hardware back-ends.

### Four fundamental challenges at the computation graph level and tensor operator level

1. **High-level dataflow rewriting:**
    - Different hardware devices may have vastly different memory hierarchies.

    - Enabling strategies to fuse operators and optimize data layouts are crucial for optimizing memory access.

2. **Memory reuse across threads:**
   - Modern GPUs and specialized accelerators ahve memory that can be shared across compute cores.
   - Traditional shared-nothing nested parallel model is no longer optimal.
   - Cooperation among threads on shared memory loaded is required for optimized kernels. 

3. **Tensorized compute intrinsics:**
   - The latest hardware provides new instructions that go beyond vector operations like the GEMM operator in TPU or the tensor core in NVIDIA's Volta.
   - Consequently, the scheduling procedure must break computation into tensor arithmetic intrinsics instead of scalar or vector code.

4. **Latency Hiding**
    - Traditional architectures with simultaneous multithreading and automatically managed caches implicitly hide latency in modern CPUs/GPUs.
    - Specialized accelerator designs favor learner control and offload most of the scheduling complexity to the compiler stack.
    - Still, scheduling must be peformed carefully to hide memory access latency.


### TVM: An End-to-End Optimization Stack

- An end-to-end optimizing compiler stack to lower and fine-tune DL workloads to diverse hardware back-ends. 
- Designed to separate:
  - the algorithm description
  - schedule
  - hardware interface
- This separation enables **support for novel specialized accelerators** and **their corresponding new intrinsics**. 
- TVM presents **two optimization layers**:
  - a computation graph optimization layer to address:
    - High-level dataflow rewriting
  - a tensor optimization layer with new schedule primitives to address:
    - memory reuse across threads
    - tensorized compute intrinsics
    - latency hiding
- 
 