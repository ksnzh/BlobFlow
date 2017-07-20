//
// Created by ksnzh on 17-7-17.
//

#ifndef BLOBFLOW_MEM_HPP
#define BLOBFLOW_MEM_HPP


#include <cstdlib>
#include <cstring>

inline void bfMalloc(void **ptr, size_t size){
    *ptr = malloc(size);
}

inline void bfFree(void *ptr){
    free(ptr);
}

inline void bfMemset(void *ptr, size_t size){
    memset(ptr, 0, size);
}

inline void bfMemcpy(void* dest, void* src, size_t size){
    memcpy(dest, src, size);
}

class Mem{
public:
    Mem() :cpu_ptr(nullptr) {}
    Mem(size_t size) :cpu_ptr(nullptr), size_(size) {}
    void to_cpu();
    const void* cpu_data();
    void set_cpu_data(void *data);
    void* mutable_cpu_data();
    void *cpu_ptr;
    size_t size_;
    bool own_cpu_data;
    ~Mem();
};

#endif //BLOBFLOW_MEM_HPP
