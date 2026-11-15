#pragma once 
#include<cstdlib>
#define MAX_PARAMS 4
#define POOL_ALLIGN 16348 // we use that number so 

struct MemPool{
    char* base;
    size_t size;
    size_t offset;
};
// mempool helper functions 
MemPool mempool_create(size_t bytes);
void* mempool_alloc(MemPool* a , size_t bytes );
float* mempool_zeroes(MemPool* a , size_t n_float); 
void mempool_reset(MemPool* a);
void mempool_delete(MemPool* a);


struct Tensor{
    float* data;
    float* grad;
    int ndim;
    size_t shape[MAX_PARAMS];
    size_t stirde[MAX_PARAMS];
    bool requires_grad;


    int op; // 0 means none 
    Tensor* src[3]; // input of the operations 
    int     iparam[4];           // op-specific ints  (e.g. B, T, n_head)
    float   fparam;              // op-specific float (e.g. layernorm eps)
    void*   saved;               // op-specific saved buffer (e.g. softmax probs)
    void*   saved2; // just in case 
};
enum OP_TYPE{
    OP_NONE = 0,
    OP_MATMUL,
    OP_ADD,
    OP_ADD_BIAS,
    OP_RELU,
    OP_GELU,
    OP_LAYERNORM,
    OP_ATTENTION,
    OP_CROSS_ENTROPY,
    OP_SUM
}; //  now we have the type of operations we can do we will need to store them as well in tensor itself
// creation (all memory from pool p)

Tensor* tensor_new  (MemPool* p, const size_t* shape, int ndim);   // zeros
Tensor* tensor_2d   (MemPool* p, size_t rows, size_t cols);        // zeros, 2-D shortcut
Tensor* tensor_full (MemPool* p, const size_t* shape, int ndim, float v);
Tensor* tensor_randn(MemPool* p, const size_t* shape, int ndim, float std, uint32_t* rng);

// info
size_t numel(const Tensor* t);
void   tensor_row_major_strides(const size_t* shape, int ndim, size_t* out_strides);

// access
float* tensor_at  (const Tensor* t, const size_t* idx); // formula is offset = sum (stridei * idxi ) and then return t->data + offset 
void   tensor_fill(Tensor* t, float v);

// views: share data, no copy. Views are NOT tracked by autograd.
Tensor* tensor_transpose(MemPool* p, const Tensor* t, int d0, int d1);
Tensor* tensor_reshape  (MemPool* p, const Tensor* t, const size_t* new_shape, int new_ndim); // gonna be a bit hard to implement 

// layout
bool    tensor_is_contiguous(const Tensor* t);
Tensor* tensor_contiguous   (MemPool* p, Tensor* t);   // returns t itself if already contiguous

// debug
void tensor_print(const Tensor* t);

// ---------------- random numbers ----------------
// xorshift32: tiny, deterministic. State must be non-zero else ub 
uint32_t rng_next   (uint32_t* s);
float    rng_uniform(uint32_t* s);   // (0, 1]
float    rng_normal (uint32_t* s);   // mean 0, std 1 (Box-Muller)