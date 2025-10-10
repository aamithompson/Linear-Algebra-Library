# Linear Algebra Library
## 1. Overview
This project is a **built from scratch** linear algebra library written for C# and Unity. It is designed to be used for **scientific computing**, **AI applications**, and **simulation environments**.

The library supports **matrix**, **vector**, and **tensor** objects with a focus on **performance**, **usability**, and **readability**.

**Key Highlights:**
- Optimized for performance with an underlying 1-D `float[]` for sequential cache access.
- Matrix multiplication optimization through loop order tuning, block tiling, pre-transposition, and parallelization.
- Intuitive, functional, and readable API for easy integration and use.

## 2. Features / Key Capabilities

### Core Data Structures

- **Vector** - 1-D arrays of arbitrary size (n)

- **Matrix** - 2-D arrays of arbitrary size (m×n)

- **Tensor** - N-D arrays for higher-order data

### Mathematical Operations

- **Element-wise operations** - Addition, subtraction, scalar multiplication, negation, hadamard product

- **Vector operations** -  Vector dot product, norm, check for orthogonality, unit vectors
  
- **Matrix operations** - Matrix multiplication, matrix-vector multiplication, transposition, determinant, norm, trace, diagonal matrices

- **Operator overloading** - Ease of use operators such as +, -, and *.

### Utility Functions

- **Constructors** - Generate zero-filled, one-filled, or randomized vectors, matrices, and tensors with support for uniform and gaussian distributions.

- **Data access** - Get, set, copy, data slicing

- **Performance hook** - Accessor for underlying data for performance-critical code

### Performance-Oriented Design
- Built with underlying 1-D array for optimized memory access

- Optimized matrix multiplication (loop order changes, pre-transpose, block tiling, parallelization)

## 3. Installation / Usage

### Installation
1. Download the repository as a ZIP or clone it:
```bash
git clone https://github.com/aamithompson/LinearAlgebraLibrary.git
```

2. Copy the `LinearAlgebraLibrary` folder into your Unity or C# project’s `Assets` or source folder.

### Usage
Once the library is in your project, you can start utilizing it like this:
```csharp
using lmath;

class Example {
    static void Main(){
          //Vectors
          //Create zero-filled, one-filled, and random vectors
          Vector a = Vector.Zeros(3);
          Vector b = Vector.Ones(3);
          Vector c = Vector.Random(Vector.Ones(3)*-1, Vector.Ones(3));

          //Vector operations
          a = (b + c) * 4;
          float val = Vector.Dot(b, c);

          //Matrices
          //Create zero-filled, one-filled, and random matrices
          Matrix A = Matrix.Zeros(3,3);
          Matrix B = Matrix.Ones(3,3);
          Matrix C = Matrix.Random(A, B);

          //Matrix Operations
          C = A + B;
          A = MatMul(B, C);
          A = Matrix.Transpose(A);
          float det = Matrix.Determinant(A);
          A = MatVecMul(A, a);
    }
}
```

## 4. Benchmark Results & Performance Results

### MATRIX MULTIPLICATION FOR 4x4 TO 16x16 MATRICES (m×n × n×p, where m,n,p ∈ [4,16])
This small range highlights the base overhead costs and effects of optimization steps.

| Step | Change | Total Time | Average Time | Speedup vs. Previous | Speedup vs. Baseline |
| - | - | - | - | - | - |
| 1 | Naive Template Implementation | 0.355s | 1.62e-04s | – | – |
| 2 | Naive Float Implementation (Baseline) | 0.052s | 2.37e-05s | 6.82x | – |
| 3 | Pre-transpose B | 0.059s | 2.67e-05s | 0.88x | 0.88x |
| 4 | Switch Index Order (i, j, k) -> (i, k, j) | 0.054s | 2.47e-05s | 1.09x | 0.95x |
| 5 | Block Tiling | 0.054s | 2.47e-05s | 1.00x | 0.95x |

**Takeaway:** Overhead costs are ultimately minimal, would probably look to remove matrix transposition from small matrices.

### MATRIX MULTIPLICATION FOR 32 x 32 to 64 x 64 MATRICES  (m×n × n×p, where m,n,p ∈ [32,64])
This range highlights practical matrix sizes for various applications.

| Step | Change | Total Time | Average Time | Speedup vs. Previous | Speedup vs. Baseline |
| - | - | - | - | - | - |
| 1 | Naive Template Implementation | 408.6s | 1.13e-2s | – | – |
| 2 | Naive Float Implementation (Baseline) | 27.37s | 7.61e-4s | 14.9x | – |
| 3 | Pre-transpose B | 28.97s | 8.06e-4s | 0.94x | 0.94x |
| 4 | Switch Index Order (i, j, k) -> (i, k, j) | 20.12s | 5.60e-4s | 1.44x | 1.36x |
| 5 | Block Tiling | 20.28s | 5.64e-4s | 0.99x | 1.35x |

**Takeaway:** Performance speedup is small with the exception of conversion from a general template matrix to floating point restriction increasing performance drastically.

### MATRIX MULTIPLICATION FOR 128 x 128 MATRIX
This matrix size is about at the size where parallelization starts to take effect.

| Step | Change | Average Time | Speedup vs. Previous | Speedup vs. Baseline |
| - | - | - | - | - |
| 1 | Naive Template Implementation | 0.177s | – | – |
| 2 | Naive Float Implementation (Baseline) | 1.01e-2s | 17.5x | – |
| 3 | Pre-transpose B | 1.03e-2s | 0.98x | 0.98x |
| 4 | Switch Index Order (i, j, k) -> (i, k, j) | 5.50e-3s | 1.87x | 1.84x |
| 5 | Block Tiling | 5.47e-3s | 1.01x | 1.85x |
| 6 | Parallelization | 4.91e-3s | 1.11x | 2.06x |

**Takeaway:** Performance after all optimization steps applied is at 2.06 x the baseline speed. Larger matrices are where the optimizations start to really shine.

### MATRIX MULTIPLICATION FOR 1024 x 1024 MATRIX
This matrix size demonstrates the scalability of optimization.

| Step | Change | Average Time | Speedup vs. Previous | Speedup vs. Baseline |
| - | - | - | - | - |
| 1 | Naive Template Implementation | 105.1s | – | – |
| 2 | Naive Float Implementation (Baseline) | 4.78s | 22.0x | – |
| 3 | Pre-transpose B | 4.261s | 1.12x | 1.12x |
| 4 | Switch Index Order (i, j, k) -> (i, k, j) | 2.92s | 1.46x | 1.63x |
| 5 | Block Tiling | 2.480s | 1.18x | 1.93x |
| 6 | Parallelization | 0.418s | 5.93x | 11.4x |

**Takeaway:** Relative performances increases and parallelization drastically boosts speed 11.4x over baseline.

## 5. Design & Implementation Notes

### API Design
- **Getters/Setters for Normal Use, Public Accessors for Performance:**
The standard `GetElement()` / `SetElement()` allow for normal, safe access to elements contained within the data objects. When a user needs to expose the underlying data for performance-critical code, the `AccessData()` function allows for direct usage. Much of the other functions within the linear math objects use this to maximize performance by avoiding repeated function calls.

- **Operator Overloading:**
Core math operations `+`, `-`, and `*` allow for intuitive and natural mathematical expression in code.

- **Static Methods:**
Much of the functions and constructors in the code are static to call back to the class for cleaner structure code. Examples of this are `Zeros()`, `Ones()`, `MatMul()`, `Determinant()`, etc.

### LArray Base Design
- **Underlying 1-D Array:**
All vector, matrix, and tensor objects are derived from the `LArray` parent class which implements a single contiguous `float[]`. The decision made for this was to minimize memory fragmentation and optimize cache usage during sequential loop patterns. The restriction to `float` was a decision to maximize performance by using a popular type for scientific computing and AI and to avoid overhead from a template based implementation.

- **General Purpose and Built for Inheritance:**
`LArray` was built with inheritance in mind and to serve as a blueprint for the vectors, matrices and tensors. The memory access and data slicing are implemented within this class as was written with N-D organization in mind. Hence we have 1-D vectors, 2-D matrices, and N-D tensors.

### Critical Performance Choices
- **Float Implementation:**
Data objects default to `float` typing for improved speed and reduce memory usage compared to template typing.

- **Loop Order Optimization**
Loop nesting order is tuned to improve cache usage.

- **Pre-transposition of B**
To reduce cache misses by making memory access patterns in the innermost loop sequential.

- **Block Tiling:**
Can switch to operate on smaller sub-blocks to improve cache reuse.

- **Parallelization:**
Large matrices have significant increase in performance by opting in parallel computation via `Parallel.For()`.

### Unity Integration
- Written for Unity by default to use in simulation environments. Has been adapted to pure C#.
 
## 6. Future Work / Optimization Considerations
While the current implementation provides a baseline performance, usage, and flexibility, there are several considerations for future development:

### Optimization
- **SMID/Vectorization** - Utilize .NET's `System.Numerics` for `Vector<T>` or other options to enable usage of multiple operations per CPU instruction, increasing GFLOPS.

- **Dynamic Optimization** - Analyze matrix size across height and length and determining transposition, operation ordering, and loop ordering to increase performance.

- **First-Time Runtime/Installation Optimization** - Benchmark the system on initialization or installation in order to determine constants for optimal block size, parallelization thresholds, and loop orders for current hardware.

- **Memory Pooling** - Reduce allocation by reusing buffers for temporary math objects in repeated calculations.

### Algorithmic Considerations
- **Sparse Matrix Support** - Implement compressed formats for operations on large but sparsely populated matrices.
  
- **Advanced Multiplication Algorithms** - While Strassen multiplication is already implemented, there is consideration for other algorithms for extremely large matrices.

### API & Usability
- **Expanded Tensor Operations** - There is basic support for tensors, but further development for mathematical functionality is considered.
  
- **Expanded Matrix Operations** - There is already extensive implementation of matrix functionality, however there are considerations for eigenvalue decomposition, LU/QR factorizations and other standard parts of matrix math.

## 7. License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
