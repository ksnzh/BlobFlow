//
// Created by ksnzh on 2017/7/17.
//

#ifndef BLOBFLOW_COMMON_HPP
#define BLOBFLOW_COMMON_HPP

#include <cstring>
#include <memory>
#include "utils/rng.hpp"
#include "utils/math.hpp"
#include <glog/logging.h>

#define NOT_IMPLEMENTED LOG(FATAL)<<"Use non-implemented codes."

class BlobFlow{
public:
    BlobFlow();
    ~BlobFlow();
    static BlobFlow& Get();
    static unsigned int get_random_seed(){ return Get().random_seed;}
    static void set_random_seed(unsigned int seed);

    static rng_t* get_rng(){
        if(!Get().random_generator){
            Get().random_generator.reset(new RNG());
        }
        rng_t* rng = Get().random_generator.get()->get_rng();
        return rng;
    }

    static unsigned  int get_random_value(){
        rng_t* rng = get_rng();
        return (*rng)();
    }
    static int64_t cluster_seedgen();
    class RNG{
    public:
        RNG() {generator.reset(new Generator());}
        RNG(unsigned int seed) {generator.reset(new Generator(seed));}
        rng_t* get_rng() {return generator.get()->get_rng();}
        class Generator{
        public:
            //pid
            Generator() :rng(new rng_t((uint32_t)BlobFlow::cluster_seedgen())) {}
            //specific
            Generator(unsigned int seed) :rng(new rng_t(seed)) {}
            rng_t* get_rng() { return rng.get();}
        private:
            std::shared_ptr<rng_t> rng;
        };
    private:
        std::shared_ptr<Generator> generator;
    };
private:
    unsigned int random_seed;
    std::shared_ptr<RNG> random_generator;
};

//模板声明与定义分离
#define INSTANTIATE_CLASS(classname) \
    template class classname<float>; \
    template class classname<double>

#endif //BLOBFLOW_COMMON_HPP
