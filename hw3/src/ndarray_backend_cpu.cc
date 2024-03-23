#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

/// needed a to be type AlignedArray* to be able to acess values at memory adress. Originally,
/// the type of a was AlignedArray& which only acessed the memory adresses NOT the values. 
void strided_index_setter(const AlignedArray* a, AlignedArray* out, std::vector<int32_t> shape,
                    std::vector<int32_t> strides, size_t offset, size_t func_case_to_run, 
                    size_t val=-1) {
  
  int depth = shape.size();
  std::vector<int32_t> indexes(depth, 0);
  int cnt = 0;
  
  /// while true is used to continue incrementing indexes until we can't anymore. 
  /// (helps do the increments and the setting all in one function)
  while (true) {
    /// offset gives how far desired element is stored in memory adress ==> index
    /// must be initialized to it. 
    int index = offset;
    for (int i = 0; i < depth; i++) {
      index += strides[i] * indexes[i];
    }

    /// updates hapening in a case by case fashion: depending in which function
    /// we are, we need to either update out or a. 
    /// 0 = "Compact", 1 = "EwiseSetitem", 2 = ""ScalarSetitem
    if (func_case_to_run == 0){
      out->ptr[cnt++] = a->ptr[index];
    }
    else if (func_case_to_run == 1){
      out->ptr[index] = a->ptr[cnt++];
    }
    else{
      out->ptr[index] = val;
    }

    // increment
    ///better way to do the incrementing that what I had originally
    indexes[depth - 1]++;

    // carry
    int idx = depth - 1;
    while (indexes[idx] == shape[idx]) {
      if (idx == 0) {
        // overflow
        return;
      }
      indexes[idx--] = 0;
      indexes[idx]++;
    }
  }
}


void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN YOUR SOLUTION
  ///&a converts AlignedArray& to AlignedArray*
  strided_index_setter(&a, out, shape, strides, offset, 0); 
  /// END YOUR SOLUTION
}


void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  strided_index_setter(&a, out, shape, strides, offset, 1);
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  strided_index_setter(NULL, out, shape, strides, offset, 2, val=val);
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
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
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  /// multiples the corresponding indexes together
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out){
  /// multiples elemets by specific val
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  ///divide elemnts of a by corresponding elements of b
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ///divide each element by a scalar
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ///bring every element to the power of val
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  ///takes the maximum of each corresponding a and b element. 
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out){
  ///takes max between element in a and val
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  /// out will be an array of 1s and 0s indicating whether corresponding elements are equal or not
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = int(a.ptr[i] == b.ptr[i]);
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out){
  /// out will be an array of 1s and 0s indicating whether elements in a are == to val
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = int(a.ptr[i] == val);
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  /// out will be an array of 1s and 0s indicating whether a.ptr[i] >= b.ptr.[i]
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = int(a.ptr[i] >= b.ptr[i]);
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out){
  /// out will be an array of 1s and 0s indicating whether a.ptr[i] >= val
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = int(a.ptr[i] >= val);
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out){
  ///takes the log of each element
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out){
  ///do e^a.ptr[i] 
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out){
  ///take the tanch of each element
  for(size_t i = 0; i < a.size; i++){
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  ///out[i,j] = out[i * cols + j], where cols = # of columns
  for(size_t i = 0; i < m; i++){
    ///p next because we need to initilize elements in out to 0 before updating 
    for(size_t j = 0; j < p; j++){
      out->ptr[i * p + j] = 0;
      for(size_t k = 0; k < n; k++){
        ///out[i,j] += a[i,k] * b[k,j]
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  ///it seems that the compiler will automatically vectorize the for loops
  /// so I guess I will just use for loops? 
  for(size_t i = 0; i < TILE; i++){
    for(size_t j = 0; j < TILE; j++){
      for(size_t k = 0; k < TILE; k++){
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**8
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  /// need to start by initializing out. Number of element is m/TILE * p/TILE * TILE * TILE = m * p.
  /// since we don't need to iterate over m and p separately like we did for matmul
  /// (we need to iterate over m/TILE, n/TILE, p/TILE, and maybe TILE), we can just have one for
  /// to do this
  for(size_t i = 0; i < m*p; i ++){
    out->ptr[i] = 0;
  }
  /// common dimension in n/TILE and the TILExTILE portion
  for(size_t j = 0; j < m/TILE; j++){
    for(size_t k = 0; k < n/TILE; k++){
      for(size_t l = 0; l < p/TILE; l++){
        /// this is how you index in 4D
        AlignedDot(&a.ptr[j * n * TILE + l * TILE * TILE], 
                   &b.ptr[l * p * TILE + k * TILE * TILE], 
                   &out->ptr[j * p * TILE + k * TILE * TILE]);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  /// in python: out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)
  /// since a.size = out.size * reduce_size ==> we did a.array[:].reshape(-1, reduce_size)
  /// step already

  /// analgous to spliting array into groups of size reduce_size and finding max of 
  /// those subarrays. 
  size_t cnt = 0;
    for(size_t i = 0; i < a.size; i += reduce_size){
        float max = a.ptr[i];
        for(size_t j = i + 1; j < i + reduce_size; j++){
            if(a.ptr[j] > max){
                max = a.ptr[j];
            }
        }
        out->ptr[cnt++] = max;
    }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  /// similar to above exept instead of find max of subarrays we find
  /// sum.
  size_t cnt = 0;
    for(size_t i = 0; i < a.size; i += reduce_size){
        float sum = 0;
        for(size_t j = i; j < i + reduce_size; j++){
           sum += a.ptr[j];
        }
        out->ptr[cnt++] = sum;
    }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
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
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}