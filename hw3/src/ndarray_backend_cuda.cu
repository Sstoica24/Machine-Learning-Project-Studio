#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256 

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  ///dim3 is standard type for dimensions. dim3 though indicates that block
  /// and grid must be in 3D
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  ///size indicated how many times out calls a.
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides
/// called from kernel so should be run on the device. 
__device__ int32_t get_index_for_a(CudaVec shape, size_t k, CudaVec strides, size_t offset){
  /// this will find the indexes array that is needed
  /// you can't redefine k because you are doing this in parallel
  /// allie said indexes array did not make sense
  /// you can't create a list in the kernel whose size is unknown to the compiler
  /// because
  size_t ind_for_a = offset;
    for(size_t i = 0; i < shape.size; i++){
        int32_t prefered_index = shape.size - 1 - i;
        ind_for_a += (k % shape.data[prefered_index]) * strides.data[prefered_index];
        k = k / shape.data[prefered_index];
    }
  return ind_for_a;
}
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    // size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    // dim.block = dim3(BASE_THREAD_NUM, 1, 1);
    // dim.grid = dim3(num_blocks, 1, 1);
    // size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index_a = get_index_for_a(shape, k, strides, offset);
  out[k] = a[index_a];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    // size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    // dim.block = dim3(BASE_THREAD_NUM, 1, 1);
    // dim.grid = dim3(num_blocks, 1, 1);
    // size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index_a = get_index_for_a(shape, k, strides, offset);
  out[index_a] = a[k];
}
  /// END SOLUTION

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

__global__ void SetItemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    // size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
    // dim.block = dim3(BASE_THREAD_NUM, 1, 1);
    // dim.grid = dim3(num_blocks, 1, 1);
    // size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t index_a = get_index_for_a(shape, k, strides, offset);
  out[index_a] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  SetItemKernel<<<dim.grid, dim.block>>>(val, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * multiply together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

///* indicates pointer ==> need to pass in a pointer to array
__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] * val;
  }
}
///& gives you the adress of the block of memory and the adress is going to have a pointer, which you
/// you can get as .ptr as that is how you defined CudaArray
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out){
  ///-> gets attribute of class
  CudaDims dims = CudaOneDim(out->size);
  ScalarMulKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] / b[index];
  }
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out){
  ///element dives two cuda arrays
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivkernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] / val;
  }
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivkernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

///* indicates pointer ==> need to pass in a pointer to array
__global__ void ScalarPowKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = pow(a[index], val);
  }
}
///& gives you the adress of the block of memory and the adress is going to have a pointer, which you
/// you can get as .ptr as that is how you defined CudaArray
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out){
  ///-> gets attribute of class
  CudaDims dims = CudaOneDim(out->size);
  ScalarPowKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    if (a[index] > b[index]){
      out[index] = a[index];
    }
    else{
      out[index] = b[index];
    }
  }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


///* indicates pointer ==> need to pass in a pointer to array
__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    if (a[index] > val){
      out[index] = a[index];
    }
    else{
      out[index] = val;
    }
  }
}
///& gives you the adress of the block of memory and the adress is going to have a pointer, which you
/// you can get as .ptr as that is how you defined CudaArray
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out){
  ///-> gets attribute of class
  CudaDims dims = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] == b[index]; 
  }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out){
  ///element dives two cuda arrays
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] == val;
  }
}
///& gives you the adress of the block of memory and the adress is going to have a pointer, which you
/// you can get as .ptr as that is how you defined CudaArray
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out){
  ///-> gets attribute of class
  CudaDims dims = CudaOneDim(out->size);
  ScalarEqKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] >= b[index]; 
  }
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out){
  ///element dives two cuda arrays
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = a[index] >= val;
  }
}
///& gives you the adress of the block of memory and the adress is going to have a pointer, which you
/// you can get as .ptr as that is how you defined CudaArray
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out){
  ///-> gets attribute of class
  CudaDims dims = CudaOneDim(out->size);
  ScalarGeKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = log(a[index]);
  }
}

void EwiseLog(const CudaArray& a, CudaArray* out){
  ///element dives two cuda arrays
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    out[index] = exp(a[index]);
  }
}

void EwiseExp(const CudaArray& a, CudaArray* out){
  ///element dives two cuda arrays
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void SingleEwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    out[index] = tanh(a[index]);
  }
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  CudaDims dim = CudaOneDim(out->size);
  SingleEwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Matr
////////////////////////////////////////////////////////////////////////////////
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
            uint32_t P){
  /// iterate over the row, columns, and the elements. To do the elements, notice that there are
  /// N elements in each row in a and N elements in each col in b ==> need to iterante over N
  /// ==> one for loop needed

  /// row index for a
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  ///col index for b
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;
  if(row < M && col < P){
    out[row * P + col] = 0;
    /// iterate over the elements
    for(size_t e = 0; e < N; e++){
      out[row * P + col] += a[row * N + e] * b[e * P + col];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  //   CudaDims CudaOneDim(size_t size) {
  // /**
  //  * Utility function to get cuda dimensions for 1D call
  //  */
  // CudaDims dim;
  /// size is how many times a is being called
  // size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  // dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  // dim.grid = dim3(num_blocks, 1, 1);
  // return dim;
  // }


  /// BEGIN SOLUTION
  ///CAN'T USE CudaOneDim LIKE ALWAYS BECAUSE WE ARE NO LONGER WORKING IN 1D!
  /// using how they defined CudaOneDim for assistance + NVIDIA
  /// grid and block should be 3D because type is dim3
 
  ///swapping grid and block worked. Not sure why.
  dim3 grid(BASE_THREAD_NUM, BASE_THREAD_NUM, 1);
  /// each row will need be called P times because there are P columns in b ==>
  /// a is read P times by each thread (one thread will calc row + col to produce
  ///one element in out). Similarly, each column in b will be read M times because
  ///there are M rows in a ==> b will be read M times. First position is for block is
  ///a, second position for block is b. We need the extra dim because type is dim3.
  dim3 block((P + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, (M + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, 1);
  ///normal calling kernel
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

// __device__ CudaArray, CudaArray, uint32_t M pad_matrices(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P){
//     ///recal A[i][j] = A[i * num_cols + j]
//     ///pad matrices. Some neet tricks I learned online
//     uint32_t padded_M = (M % TILE == 0) ? M : M + (TILE - M % TILE);
//     uint32_t padded_N = (N % TILE == 0) ? N : N + (TILE - N % TILE);
//     uint32_t padded_P = (P % TILE == 0) ? P : P + (TILE - P % TILE);

//     std::vector<int32_t> paddedMatrix_a(padded_M * padded_N);
//     std::vector<int32_t> paddedMatrix_b(padded_N * padded_P);
//     for(size_t i = 0; i < m; i ++){
//         std::copy(arr.begin() + i * n, arr.begin() + (i + 1) * n, paddedMatrix_a.begin() + i * padded_n);
//         std::copy(arr.begin() + i * n, arr.begin() + (i + 1) * n, paddedMatrix_b.begin() + i * padded_n);
//     }

//     return paddedMatrix_a, paddedMatrix_b, padded_N;
// }
// __global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N,
//             uint32_t P){
//       CudaArray VecToCuda(a), CudaArray VecToCuda(b), padded_P = pad_matrices(a, b, M, N, P);
//       ///two statitacly sized pieces of memory. This will be the considered regions in a and b
//       /// respectivly of size TILE x TILE. A is row wise B is column wise
//       /// completed with the help of NVIDIA, lecture 11
//       __shared__ int32_t A[TILE * TILE];
//       __shared__ int32_t B[TILE * TILE];

//       /// block indexes
//       size_t bx = blockIdx.x;
//       size_t by = blockIdx.y;
//       // Thread row and column within out
//       size_t tx = threadIdx.x;
//       size_t ty = threadIdx.y;

//       ///global row and col for the thread
//       /// need to count # of cols to get # of elements in row ==> use by and ty
//       int row = by * TILE + ty;
//       /// need to count # of rows to get # of elements in col ==> bx and tx
//       int col = bx * TILE + tx;

//       ///get desired element in sub TILE x TILE matrices
//       ///like before, we iterate over # of elements in rows and cols
//       for(size_t i = 0; i < padded_N/TILE; i++){
//         /// for right side: row * padded_N gives current row, i * TILE indexes the set
//         /// of columns we are working on and tx increments within that column
//         A[ty * TILE + tx] = a[row * padded_N + (i * TILE + tx)];
//         /// i * tile_size * padded_N index finds the row we are considering (where the element)
//         /// is. ty * padded_N gives the sepecific index of desired element since padded_N elements
//         /// in each column, so finding next element in col, you would need to skip padded_N elements
//         /// the row. Then col gives us the col that we want. 
//         B[ty * TILE + tx] = b[(i * tile_size * padded_N + ty*padded_N) + col];
//       }
//       ///synchronize threads to make sure all threads finish (need to always do this after an operation)
//       __syncthreads();
//       ///now multiply A[ty * TILE + tx] and B[ty * TILE + tx] together to get value. To do this iterate over j
//       int32_t sum = 0;
//       for(size_t j = 0; j < TILE; j++){
//         /// in NVIDIA documentation, this is Cvalue += As[row][e] * Bs[e][col];
//         ///row is constant because we iterate through the columns in A
//         /// col is constant because we loop through the rows in B
//         sum += A[ty * TILE + j] * B[j * TILE + tx];
//       }
//       /// performed operation, so make sure threads are synched
//       __syncthreads();

//       ///then normally update out
//       out[row * padded_N + col] = sum;
// }

// void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
//             uint32_t P) {
//   /**
//    * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
//    * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
//    * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
//    * over (i,j) entries in the output array.  However, to really get the full benefit of this
//    * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
//    * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
//    * the CPU backend, here you should implement a single function that works across all size
//    * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
//    * implementations, this function here will largely just set up the kernel call, and you should
//    * implement the logic in a separate MatmulKernel() call.
//    * 
//    *
//    * Args:
//    *   a: compact 2D array of size m x n
//    *   b: comapct 2D array of size n x p
//    *   out: compact 2D array of size m x p to write the output to
//    *   M: rows of a / out
//    *   N: columns of a / rows of b
//    *   P: columns of b / out
//    */

//   //   CudaDims CudaOneDim(size_t size) {
//   // /**
//   //  * Utility function to get cuda dimensions for 1D call
//   //  */
//   // CudaDims dim;
//   /// size is how many times a is being called
//   // size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
//   // dim.block = dim3(BASE_THREAD_NUM, 1, 1);
//   // dim.grid = dim3(num_blocks, 1, 1);
//   // return dim;
//   // }


//   /// BEGIN SOLUTION
//   ///CAN'T USE CudaOneDim LIKE ALWAYS BECAUSE WE ARE NO LONGER WORKING IN 1D!
//   /// using how they defined CudaOneDim for assistance + NVIDIA
//   /// grid and block should be 3D because type is dim3
 
//   ///swapping grid and block worked. Not sure why.
//   dim3 grid(BASE_THREAD_NUM, BASE_THREAD_NUM, 1);
//   /// each row will need be called P times because there are P columns in b ==>
//   /// a is read P times by each thread (one thread will calc row + col to produce
//   ///one element in out). Similarly, each column in b will be read M times because
//   ///there are M rows in a ==> b will be read M times. First position is for block is
//   ///a, second position for block is b. We need the extra dim because type is dim3.
//   dim3 block((P + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, (M + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, 1);
//   ///normal calling kernel
//   MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
//   /// END SOLUTION
// }


////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t offset = index * reduce_size;
  for(size_t i = 0; i < reduce_size; i++){
    out[index] += a[i + offset];
    }
}

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size){
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index < size){
    ///this will give the next index that we want in a (called offset because we need to skip by 
    ///reduce sum each time
    size_t offset = index * reduce_size;
    out[index] = a[offset];
    /// for loops are fine in cuda functions
    for(size_t i = 1; i < reduce_size; i++){
      if(a[i + offset] > out[index]){
        out[index] = a[i + offset];
      }
    }
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}



void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh); 

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
