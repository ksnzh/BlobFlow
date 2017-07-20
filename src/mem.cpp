//
// Created by ksnzh on 17-7-17.
//

#include "mem.hpp"

void Mem::to_cpu() {
    bfMalloc(&cpu_ptr, size_);
    bfMemset(cpu_ptr, size_);
    own_cpu_data = true;
}

const void* Mem::cpu_data() {
    to_cpu();
    return (const void*)cpu_ptr;
}

void Mem::set_cpu_data(void *data) {
    if(own_cpu_data){
        bfFree(cpu_ptr);
    }
    cpu_ptr = data;
    own_cpu_data = false;
}

void *Mem::mutable_cpu_data() {
    to_cpu();
    return cpu_ptr;
}

Mem::~Mem() {
    if(cpu_ptr && own_cpu_data) bfFree(cpu_ptr);
}