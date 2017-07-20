//
// Created by ksnzh on 17-7-17.
//

#include "common.hpp"
#include <boost/thread.hpp>

static boost::thread_specific_ptr<BlobFlow> thread_instance;

BlobFlow& BlobFlow::Get() {
    if(!thread_instance.get()){
        thread_instance.reset(new BlobFlow());
    }
    return *(thread_instance.get());
}

int64_t BlobFlow::cluster_seedgen() {
    int64_t seed, pid, t;
    pid = getpid();
    t = time(0);
    seed = abs(((t * 181) *((pid - 83) * 359)) % 104729); //set casually
    return seed;
}

BlobFlow::BlobFlow() {

}

BlobFlow::~BlobFlow() {

}

void BlobFlow::set_random_seed(unsigned int seed) {
    Get().random_seed = seed;
    Get().random_generator.reset(new RNG(seed));
}

