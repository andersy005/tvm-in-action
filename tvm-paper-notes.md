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

## Optimizing Computational Graphs

### Computational Graph

- Computational graphs are a common way to represent programs in DL frameworks. 
- They provide a global view on computation tasks, yet avoid specifying how each computation task needs to be implemented. 



### Operator Fusion

- An optimization that can greatly reduce execution time, particulary in GPUs and specialized accelerators.
- The idea is to **combine multiple operators together into a single kernel without saving the intermediate results back into global memory**
 
![](https://i.imgur.com/mlNhoDT.png)

**Four categories of graph operators**:

- Injective (one-to-one map)
- Reduction
- Complex-out-fusable (can fuse element-wise map to output)
- Opaque (cannot be fused)

![](https://i.imgur.com/XnhSWVN.png)

### Data Layout Transformation

- Tensor operations are the basic operators of computational graphs
- They can have divergent layout requirements across different operations
- Optimizing data layout starts with specifying the preferred data layout of each operator given the constraints dictating their implementation in hardware.

![](https://i.imgur.com/0J5QxGs.png)

### Limitations of Graph-Level Optimizations

- They are only as effective as what the operator library provides.
- Currently, the few DL frameworks that support operator fusion require the operator library to provide an implementation of the fused patterns.
    - With more network operators introduced on a regular basis, this approach is no longer sustainable when targeting an increasing number of hardware back-ends.
- It is not feasible to handcraft operator kernels for this massive space of back-end specific operators
    - TVM provides a code-generation approach that can generate tensor operators. 

## Optimizing Tensor Operations

### Tensor Expression Language

- TVM introduces a dataflow tensor expression language to support automatic code generation.
- Unlike high-level computation graph languages, where the implementation of tensor operations is opaque, *each operation is described in an index formula expression language*.

![](https://i.imgur.com/LG1pguT.png)

- TVM tensor expression language supports common arithmetic and math operations found in common language like C. 
- TVM explicitly introduces a **commutative reduction** operator to easily schedule commutative reductions across multiple threads. 
- TVM further introduces a **high-order scan operator** that can combine basic compute operators to form recurrent computations over time. 

### Schedule Space 

- Given a tensor expression, it is challenging to create high-performance implementations for each hardware back-end. 
- Each optimized low-level program is the result of different combinations of scheduling strategies, imposing a large burden on the kernel writer.
- TVM adopts the **principle of decoupling compute descriptions from schedule optimizations**.
- Schedules are the specific rules that lower compute descriptions down to back-end-optimized implementations. 

![](https://i.imgur.com/JUikGQz.png)

![](https://i.imgur.com/BCg6gCz.png)


### Nested Parallelism with Cooperation

- Parallel programming is key to improving the efficiency of compute intensive kernels in deep learning workloads. 
- Modern GPUs offer massive parallelism 
    
    ---> Requiring TVM to bake parallel programming models into schedule transformations

- Most existing solutions adopt a parallel programming model referred to as [nested parallel programs](https://youtu.be/4lS_WThsFoM), which is a form of [fork-join parallelism](https://en.wikipedia.org/wiki/Fork%E2%80%93join_model). 
- TVM uses a parallel schedule primitive to parallelize a data parallel task
  - Each parallel task can be further recursively subdivided into subtasks to exploit the multi-level thread hierarchy on the target architecture (e.g, thread groups in GPU)
- This model is called **shared-nothing nested parallelism**
  - One working thread cannot look at the data of its sibling within the same parallel computation stage.
  - Interactions between sibling threads happen at the join stage, when the subtasks are done and the next stage can consume the data produced by the previous stage. 
  - This programming model **does not enable threads to cooperate with each other in order to perform collective task within the same parallel stage**.

- A better alternative to the shared-nothing approach is to **fetch data cooperatively across threads**
    - This pattern is well known in GPU programming using languages like CUDA, OpenCL and Metal.
    - **It has not been implemented into a schedule primitive.**
- TVM introduces the **concept of memory scopes to the schedule space**, so that a stage can be marked as shared.
    - Without memory scopes, automatic scope inference will mark the relevant stage as thread-local.
    - Memory scopes are useful to GPUs.
    - Memory scopes allow us to tag special memory buffers and create special lowering rules when targeting specialized deep learning accelerators. 

![](https://i.imgur.com/HHYtujL.png)


### Tensorization: Generalizing the Hardware Interface

- **Tensorization** problem is analogous to the **vectorization** problem for [SIMD architectures](https://en.wikipedia.org/wiki/SIMD). 
- Tensorization differs significantly from vectorization
    - The inputs to the tensor compute primitives are multi-dimensional, with fixed or variable lengths, and dictate different data layouts.
    - Cannot resort to a fixed set of primitives, as new DL accelerators are emerging with their own flavors of tensor instructions. 
- To solve this challenge, TVM **separates the hardware interface from the schedule**:
    - TVM introduces a tensor intrinsic declaration mechanism
    - TVM uses the tensor expression language to declare the behavior of each new hardware intrinsic, as well as the lowering rule associated to it. 
    - TVM introduces a **tensorize** schedule primitive to replace a unit of computation with the corresponding tensor intrinsics. 
    - The compiler matches the computation pattern with a hardware declaration, and lowers it to the corresping hardware intrinsic. 
   

### Compiler Support for Latency Hiding

- **Latency Hiding:** refers to the process of overlapping memory operations with computation to maximize memory and compute utilization. 
- It requires different different strategies depending on the hardware back-end that is being targeted. 
- On CPUs, memory latency hiding is achieved **implicitly with simultaneous multithreading** or **hardware prefetching techniques**. 
- GPUs rely on **rapid context switching of many wraps of threads** to maximize the utilization of functional units. 
- TVM provides a virtual threading schedule primitive that lets the programmer specify a high-level data parallel program that TVM automatically lowers to a low-level explicit data dependence program. 


## Code Generation and Runtime Support 

### Code Generation

- For a specific tuple of data-flow declaration, axis relation hyper-graph, and schedule tree, TVM can generate lowered code by:
  - iteratively traversing the schedule tree
  - inferring the dependent bounds of the input tensors (using the axis relation hyergraph)
  - generating the loop nest in the low-level code
- The code is lowered to an in-memory representation of an imperative C style loop program. 
- TVM reuses a variant of Halide's the loop program data structure in this process. 
- TVM reuses passes from Halide for common lowering primitives like storage flattening and unrolling, 
  - and add GPU/accelerator-specific transformations such as:
    - *synchronization point detection*
    - *virtual thread injection**
    - *module generation*
- Finally, the loop program is transformed into **LLVM** or **CUDA/Metal/OpenCL** source code.

### Runtime Support

- For GPU programs, TVM builds the host and device modules **separately** and provide a runtime module system that launch kernels using corresponding driver APIs. 

### Remote Deployment Profiling

- TVM includes infrastructure to make profiling and autotuning easier on embedded devices. 
- Traditionally, targeting an embedded device for tuning requires:
  - cross-compiling on the host side, 
  - copying to the target device, 
  - and timing the execution

- TVM provides remote function call support. Through the **RPC interface**:
  - TVM compiles the program on a host compiler
  - it uploads to remote embedded devices
  - it runs the funcion remotely, 
  - and it accesses the results in the same script on the host. 

![](https://i.imgur.com/oL0Z9pp.png)


## Conclusion

- TVM provides an end-to-end stack to solve fundamental optimization challenges across a diverse set of hardware back-ends.
- TVM can encourage more studies of programming languages, compilation, and open new opportunities for hardware co-design techniques for deep learning systems. 

