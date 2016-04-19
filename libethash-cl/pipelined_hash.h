// Pipelined hashing by Optiminer.
// Please send ETH donations to 72afb263c4402ce70b17ce36c991ef3674e47f35

#ifndef PIPELINED_HASH
#define PIPELINED_HASH

#include <CL/cl.hpp>

#include <utility>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>
#include <memory>
#include <cmath>
#include <algorithm>

typedef union {
    ulong ulongs[128 / sizeof(ulong)];  // 16
    uint uints[128 / sizeof(uint)];     // 32
} hash128_t; // 64 bytes

typedef union {
    ulong ulongs[64 / sizeof(ulong)];  // 8
    uint uints[64 / sizeof(uint)];     // 16
} hash64_t; // 64 bytes

typedef union {
    ulong ulongs[32 / sizeof(ulong)];  // 4
    uint uints[32 / sizeof(uint)];     // 8
} hash32_t; // 32 bytes

struct Buffers {
    cl::Buffer lookup_data;
    cl::Buffer header;

    cl::Buffer hash32[2];
    cl::Buffer hash64[2];
    cl::Buffer output[2];
};

class PipelinedHash {

public:
    class Listener {
    public:
        virtual ~Listener() {
        }

        // reports progress, return true to abort
        virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
        virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
    };


    PipelinedHash();

    ~PipelinedHash() {
        delete[] output;
    }

    void init(cl::Context& context, cl::Device& device) {
        this->context = context;
        this->device = device;
    }

    void setDAG(hash128_t* dag, size_t size);

    void setWork(hash32_t* header, ulong target);

    void setEnableProfiling(bool enable) {
        this->profiling = enable;
    }

    bool selfCheck();

    void run(Listener* listener, ulong start_nonce);

private:
    void init_kernels();
    void init_buffers(Buffers& buffers, ulong items_per_loop);
    void setArgsInit(cl::Buffer& header, ulong start_nonce, cl::Buffer hash64);
    void setArgsMix(cl::Buffer lookup_data, cl::Buffer hash64, cl::Buffer hash32);
    void setArgsFinal(ulong target, cl::Buffer& hash64, cl::Buffer& hash32, cl::Buffer& output);

    static void printStats(cl_event event, int unused, void* kernel_);

    cl::Context context;
    cl::Device device;

    cl::Kernel hash_init;
    cl::Kernel hash_mix;
    cl::Kernel hash_final;

    cl::CommandQueue queue_dag;

    Buffers buffers;

    ulong items_per_loop = 1024 * 1024 * 4;

    ulong target = 0;

    hash128_t* dag = 0;
    size_t dag_size = 0;

    uint* output;

    bool profiling = false;
};


#endif
