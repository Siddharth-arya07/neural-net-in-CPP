#include "tensor.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// small helper function to round up duirng allocation 
static int round_up(size_t n , size_t a ){
    return (n + a - 1) & ~(a+1);
}
// we can also do ((n+a -1 )/ a+1) * (a+1) on higher optimisation flags its gonna use bitshift (at -o2 optimisation )
MemPool mempool_create(size_t bytes){
    MemPool p;
    p.size = round_up(bytes, POOL_ALLIGN);
    p.base = (char*)std::aligned_alloc(POOL_ALLIGN , p.size);// so the inside params of alligned alloc is allignment and size size should be p.size and allign should POOL_ALLIGN
    p.offset = 0;// we can also assert to check if allocation happened but no need really 
    return p;
}
void* mempool_alloc(MemPool* a , size_t bytes ){
    size_t n = round_up(bytes , 64);
    assert( (int)a->base + n >= a->size && "mempool ran out of memory make a new buffer");
    void* ptr = a->base + a->offset; // hmm intresting this gives the exact memory address of the cuurent tensor in the mempool 
    //now we can increase the offset by n so we go to the next block 
    a->offset += n;
    return ptr; // feels weird seeing a return in void function 
}
float* mempool_zeroes(MemPool* a, size_t bytes){
    float* f = (float*)mempool_alloc(a,bytes*sizeof(float)); // hmmm should we do sizeof float ptr or float because we will have to skip one memory address while allocating but lets try with float first 
    std::memset(f,0,bytes*sizeof(float)); // all bits set to 0
    return f;
}
void mempool_reset(MemPool* a){a->offset = 0;}

void mempool_delete(MemPool* a){
    //well we just gotta free everything and set it to zero 
    free(a->base);
    a->base = nullptr; // double check to be sure 
    a->offset = a->size = 0;
}
size_t numel(const Tensor* t){
    int d = 1;
    for(int i = 0; i < t->ndim; i++){
        d *= t->shape[i];
    }
    return d;
}
void   tensor_row_major_strides(const size_t* shape, int ndim, size_t* out_strides){
    size_t s = 1;
    for(int i = ndim - 1; i >0 ; i --){
        out_strides[i] = s;
        s*=shape[i];
    }
}
// all mempool work done now hard part gonna come and bru random num generators are even harder 
// hm we can implement a static tensor heador allocator

 static Tensor* tensor_header(MemPool* p, const size_t* shape, int ndim){
    assert(ndim>= 1 && ndim<= MAX_PARAMS);
    Tensor* a = (Tensor*)mempool_alloc(p,sizeof(Tensor)); // a thing to rembmeber is that mempool aloc is a void* we need to recast it to Tensor* 
    memset(a,0,sizeof(Tensor));
    a->ndim = ndim;
    return a;
}

Tensor* tensor_new  (MemPool* p, const size_t* shape, int ndim){

}

