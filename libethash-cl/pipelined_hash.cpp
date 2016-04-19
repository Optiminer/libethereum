#include "pipelined_hash.h"

#include <functional>
#include <sstream>

#include "ethash_cl_miner_kernel.h"

using std::vector;

namespace {

#define MAX_OUTPUTS 100
#define SUBGROUPS 2
#define THREADS_PER_ITEM 32

struct PrintStatsData {
    PrintStatsData(cl::Kernel* kernel, ulong items)
    : kernel(kernel),
      items(items) {}

    cl::Kernel* kernel;
    ulong items;
};

inline void checkErr(cl_int err, std::string name) {
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template<typename T>
bool verify_checksum(T* buffer, int items, ulong expected_checksum, std::string name) {
    ulong checksum = 0;
    int longs = sizeof(T) / sizeof(ulong);
    for (int i = 0; i < items; ++i) {
        for (int j = 0; j < longs; ++j) {
            checksum ^= ((ulong*)(&buffer[i]))[j];
        }
    }
    if (checksum != expected_checksum) {
        std::cout << name << " check sum mismatch: " << checksum
                << std::endl;
        for (int i = 0; i < items; ++i) {
            std::cout << i << ": ";
            for (int j = 0; j < longs; ++j) {
                std::cout << ((ulong*)(&buffer[i]))[j] << " ";
            }
            std::cout << std::endl;
        }
        return false;
    }
    return true;
}

long get_current_time_us (void)
{
    long            us;
    time_t          s;
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    us = round(spec.tv_nsec / 1e3);

    return s * 1e6 + us;
}

} // namespace


// PipelinedHash

void PipelinedHash::setArgsInit(cl::Buffer& header, ulong start_nonce, cl::Buffer hash64) {
    cl_int err = hash_init.setArg(0, header);
    checkErr(err, "Kernel::setArg(0)");
    err = hash_init.setArg(1, start_nonce);
    checkErr(err, "Kernel::setArg(1)");
    err = hash_init.setArg(2, hash64);
    checkErr(err, "Kernel::setArg(2)");
}

void PipelinedHash::setArgsMix(cl::Buffer lookup_data, cl::Buffer hash64,
        cl::Buffer hash32) {
    cl_int err = hash_mix.setArg(0, lookup_data);
    checkErr(err, "Kernel::setArg(0)");
    err = hash_mix.setArg(1, hash64);
    checkErr(err, "Kernel::setArg(1)");
    err = hash_mix.setArg(2, hash32);
    checkErr(err, "Kernel::setArg(2)");
}

void PipelinedHash::setArgsFinal(ulong target, cl::Buffer& hash64, cl::Buffer& hash32,
        cl::Buffer& output) {
    cl_int err = hash_final.setArg(0, target);
    checkErr(err, "Kernel::setArg(0)");
    err = hash_final.setArg(1, hash64);
    checkErr(err, "Kernel::setArg(1)");
    err = hash_final.setArg(2, hash32);
    checkErr(err, "Kernel::setArg(2)");
    err = hash_final.setArg(3, output);
    checkErr(err, "Kernel::setArg(3)");
}


void PipelinedHash::init_kernels() {
    if (dag_size == 0) {
        checkErr(-1, "dag size not set");
    }

    std::stringstream ss;
    ss << "-DDAG_SIZE=" << dag_size << " ";
    ss << "-DMAX_OUTPUTS=" << MAX_OUTPUTS << " ";
    ss << "-DSUBGROUPS=" << SUBGROUPS << " ";
    std::string dagSizeSource = ss.str();

    cl::Program::Sources sources;

    std::string prog(ETHASH_CL_MINER_KERNEL, ETHASH_CL_MINER_KERNEL + ETHASH_CL_MINER_KERNEL_SIZE);
    sources.push_back(std::make_pair(prog.c_str(), prog.length()));

    vector<cl::Device> devices;
    devices.push_back(device);

    cl::Program program(context, sources);
    cl_int err = program.build(devices, dagSizeSource.c_str());
    if (err != CL_SUCCESS) {
        std::cout << "Build Status: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0])
                << std::endl;
        std::cout << "Build Options:\t"
                << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0])
                << std::endl;
        std::cout << "Build Log:\t "
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                << std::endl;
    }
    checkErr(err, "Program::build()");

    const char* kernel_hash_init_name = "hash_init";
    const char* kernel_mix_name = "compute_hash_simple";
    const char* kernel_hash_final_name = "hash_final";

    hash_init = cl::Kernel(program, kernel_hash_init_name, &err);
    checkErr(err, "Kernel::Kernel()");

    hash_final = cl::Kernel(program, kernel_hash_final_name, &err);
    checkErr(err, "Kernel::Kernel()");

    hash_mix = cl::Kernel(program, kernel_mix_name, &err);
    checkErr(err, "Kernel::Kernel()");
}


bool PipelinedHash::selfCheck() {
    std::cout << "Starting GPU self check" << std::endl;
    bool pass = true;

    long lookup_size = 1024 * 1024 * 1024;
    ulong start_nonce = 10000;
    long test_items = 1024 * 3;
    //long test_items = 64;

    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

    int* lookup_data = new int[lookup_size / sizeof(int)];
    for (unsigned int i = 0; i < lookup_size / sizeof(int); ++i) {
        lookup_data[i] = i;
    }
    setDAG((hash128_t*)lookup_data, lookup_size / sizeof(hash128_t));

    // Check hash_init kernel:
    cl_int err;
    cl::CommandQueue queue(context, devices[0]);

    hash32_t header;
    for(int i=0; i<8; ++i) {
        header.uints[i] = 100000 * i;
    }
    err = queue.enqueueWriteBuffer(buffers.header, CL_TRUE, 0, sizeof(header), &header, NULL, NULL);
    checkErr(err, "CommandQueue::enqueueWriteBuffer(buf_header)");

    setArgsInit(buffers.header, start_nonce, buffers.hash64[0]);

    err = queue.enqueueNDRangeKernel(hash_init,
            cl::NullRange,
            cl::NDRange(test_items, 1),
            cl::NDRange(64, 1),
            NULL, NULL);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_hash_init)");

    err = queue.finish();
    checkErr(err, "hash_init finish");

    hash64_t* hash64 = new hash64_t[test_items];
    err = queue.enqueueReadBuffer(buffers.hash64[0], CL_TRUE, 0, test_items * sizeof(hash64_t), hash64, NULL, NULL);
    pass |= verify_checksum(hash64, test_items, 11763506336122282995UL, "hash_init kernel");
    delete[] hash64;

    // Check hash_mix kernel:

    err = queue.enqueueWriteBuffer(buffers.lookup_data, CL_TRUE, 0, lookup_size, lookup_data, NULL, NULL);
    checkErr(err, "CommandQueue::enqueueWriteBuffer(lookup_data_cl)");

    setArgsMix(buffers.lookup_data, buffers.hash64[0], buffers.hash32[0]);

    err = queue.enqueueNDRangeKernel(hash_mix,
                  cl::NullRange,
                  cl::NDRange(THREADS_PER_ITEM * test_items, 1),
                  cl::NDRange(SUBGROUPS * THREADS_PER_ITEM, 1),
                  NULL, NULL);
    checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_mix)");

    err = queue.finish();
    checkErr(err, "hash_mix finish");

    hash32_t* hash32 = new hash32_t[test_items];
    err = queue.enqueueReadBuffer(buffers.hash32[0], CL_TRUE, 0, test_items * sizeof(hash32_t), hash32, NULL, NULL);
    pass |= verify_checksum(hash32, test_items, 15870579533788283648UL, "hash_mix kernel");
    delete[] hash32;

    // Check hash_final kernel:

    uint* output = new uint[MAX_OUTPUTS + 1];
    for(int i = 0; i< MAX_OUTPUTS +1 ; ++i) {
        output[i] = 0;
    }

    err = queue.enqueueWriteBuffer(buffers.output[0], CL_TRUE, 0, sizeof(uint), output);
    checkErr(err, "ComamndQueue::enqueueWriteBuffer()");

    ulong target = 1L<<(64 - 6);
    setArgsFinal(target, buffers.hash64[0], buffers.hash32[0], buffers.output[0]);

    err = queue.enqueueNDRangeKernel(hash_final, cl::NullRange,
            cl::NDRange(test_items, 1), cl::NDRange(64, 1));
    checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_hash_final)");

    err = queue.finish();
    checkErr(err, "hash_final finish");

    err = queue.enqueueReadBuffer(buffers.output[0], CL_TRUE, 0,
            MAX_OUTPUTS * sizeof(uint), output);
    checkErr(err, "ComamndQueue::enqueueReadBuffer()");


    uint reported_matches = std::min((uint)MAX_OUTPUTS, output[0]);
    std::sort(output + 1, output + reported_matches + 1);
    ulong checksum = 0;
    for (uint i = 0; i < reported_matches; ++i) {
        ulong nonce = start_nonce + output[i + 1];
        checksum = 3 * checksum + nonce;
    }

    if (checksum != 14910974536423433315UL) {
        std::cout << "found " << output[0] << " matches with checksum "
                << checksum << std::endl;
        for (uint i = 0; i < reported_matches; ++i) {
            std::cout << output[i + 1] << std::endl;
        }
        pass = false;
    }

    delete[] output;
    delete[] lookup_data;

    std::cout << "self-check " << (pass ? "passed" : "failed") << std::endl;

    return pass;
}

void PipelinedHash::setWork(hash32_t* header, ulong target) {
    this->target = target;

    cl_int err;
    err = queue_dag.enqueueWriteBuffer(buffers.header, CL_FALSE, 0, sizeof(hash32_t), header, NULL, NULL);
    checkErr(err, "CommandQueue::enqueueWriteBuffer(header)");
}

void PipelinedHash::setDAG(hash128_t* dag, size_t size) {
    std::cout << "DAG has " << size << " elements resulting in a size of " << size * sizeof(hash128_t) << "bytes \n";
    this->dag_size = size;

    init_kernels();
    init_buffers(buffers, items_per_loop);

    cl_int err;
    queue_dag = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkErr(err, "CommandQueue::CommandQueue()");

    err = queue_dag.enqueueWriteBuffer(buffers.lookup_data, CL_FALSE, 0, size * sizeof(hash128_t), dag, NULL, NULL);
    checkErr(err, "CommandQueue::enqueueWriteBuffer(dag)");
}

void PipelinedHash::init_buffers(Buffers& buf, ulong items_per_loop) {
    cl_int err;
    buf.lookup_data = cl::Buffer(context, CL_MEM_READ_ONLY, dag_size * sizeof(hash128_t), NULL, &err);
    checkErr(err, "buf_lookup_data");

    buf.header = cl::Buffer(context, CL_MEM_READ_ONLY, 32, NULL, &err);
    checkErr(err, "buf_header");

    for (int i = 0; i < 2; ++i) {
        buf.hash64[i] = cl::Buffer(context, CL_MEM_READ_WRITE, items_per_loop * sizeof(hash64_t), NULL, &err);
        checkErr(err, "buf_hash64");

        buf.hash32[i] = cl::Buffer(context, CL_MEM_READ_WRITE, items_per_loop * sizeof(hash32_t), NULL, &err);
        checkErr(err, "buf_hash32");

        buf.output[i] = cl::Buffer(context, CL_MEM_READ_WRITE, (MAX_OUTPUTS + 1) * sizeof(uint), NULL, &err);
        checkErr(err, "buf_output");
    }
}

void PipelinedHash::run(Listener* listener, ulong start_nonce) {
    cl_int err;

    cl::CommandQueue queue_mix(context, device, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");

    cl::CommandQueue queue_hash(context, device, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    checkErr(err, "CommandQueue::CommandQueue()");

    queue_dag.finish();

    queue_mix.enqueueBarrier(); // TODO remove
    queue_hash.enqueueBarrier();

    int threads_per_item = THREADS_PER_ITEM;
    int buffer = 0;

    uint* zero = new uint[1];
    zero[0] = 0;

    long loop_start_time_us = get_current_time_us();

    bool exit = false;

    cl::Event event_mix;
    uint loop = 0;
    for (; !exit; ++loop) {
        int hash_buffer = buffer;

        cl::Event event_hash_final;
        cl::Event event_copy_output;
        if (loop > 1) {
            // Enqueue clearing result count:
            err = queue_hash.enqueueWriteBuffer(buffers.output[hash_buffer], CL_TRUE, 0, sizeof(uint), zero, NULL, NULL);
            checkErr(err, "ComamndQueue::enqueueWriteBuffer()");

            queue_hash.enqueueBarrier();

            setArgsFinal(target,
                    buffers.hash64[hash_buffer],
                    buffers.hash32[hash_buffer],
                    buffers.output[hash_buffer]);

            std::vector<cl::Event> mix_events;
            mix_events.push_back(event_mix);
            // Enqueue final hash kernel:
            err = queue_hash.enqueueNDRangeKernel(hash_final,
                    cl::NullRange,
                    cl::NDRange(items_per_loop, 1),
                    cl::NDRange(64, 1),
                    &mix_events, &event_hash_final);
            checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_hash_final)");

            // Enqueue copy output buffer back:
            err = queue_hash.enqueueReadBuffer(buffers.output[hash_buffer], CL_FALSE, 0, (MAX_OUTPUTS + 1) * sizeof(uint), output, NULL, &event_copy_output);
            checkErr(err, "ComamndQueue::enqueueReadBuffer()");
        }

        setArgsInit(buffers.header, start_nonce, buffers.hash64[hash_buffer]);

        // Enqueue hash init kernel:
        cl::Event event_hash_init;
        err = queue_hash.enqueueNDRangeKernel(
                hash_init,
                cl::NullRange,
                cl::NDRange(items_per_loop, 1),
                cl::NDRange(64, 1),
                NULL, &event_hash_init);
        checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_hash_init)");

        if (profiling) {
            event_hash_init.setCallback(CL_COMPLETE, printStats, new PrintStatsData(&hash_init, items_per_loop));
        }

        int mix_buffer = (buffer + 1) % 2;

        if (loop > 0) {
            setArgsMix(buffers.lookup_data, buffers.hash64[mix_buffer], buffers.hash32[mix_buffer]);

            // Enqueue mix kernel:
            err = queue_mix.enqueueNDRangeKernel(
                    hash_mix,
                    cl::NullRange,
                    cl::NDRange(threads_per_item * items_per_loop, 1),
                    cl::NDRange(SUBGROUPS * THREADS_PER_ITEM, 1),
                    NULL, &event_mix);
            checkErr(err, "CommandQueue::enqueueNDRangeKernel(kernel_mix)");

            if (profiling) {
                event_mix.setCallback(CL_COMPLETE, printStats, new PrintStatsData(&hash_mix, items_per_loop));
            }
        }

        if (loop > 1) {
            event_copy_output.wait();
            // Process output:
            if (output[0] > 0) {
                uint found_size = std::min((uint) MAX_OUTPUTS, output[0]);

                ulong found[MAX_OUTPUTS];
                for (uint i = 0; i < found_size; ++i) {
                    ulong nonce = output[i + 1] + start_nonce - 2 * items_per_loop;
                    found[i] = nonce;
                }
                exit |= listener->found(found, found_size);
            }
            exit |= listener->searched(start_nonce - 2 * items_per_loop, items_per_loop);
        }

        buffer = (buffer + 1) % 2;
        start_nonce += items_per_loop;
    }

    long loop_time_us = get_current_time_us() - loop_start_time_us;
    long items = items_per_loop * (loop - 1);

    queue_hash.finish();
    queue_mix.finish();

    if (profiling) {
      std::cout << "Processed a total of " << items << " items in " << loop_time_us << "us = " << (1000000.0 * items / loop_time_us) << " items/s\n";
    }

}

PipelinedHash::PipelinedHash() {
    output = new uint[MAX_OUTPUTS + 1];
    std::cout << "*************************************************************************" << std::endl;
    std::cout << "* Using PipelinedHash OpenCL kernel by Optiminer                        *" << std::endl;
    std::cout << "* Please send ETH donations to 72afb263c4402ce70b17ce36c991ef3674e47f35 *" << std::endl;
    std::cout << "*************************************************************************" << std::endl;
}

void PipelinedHash::printStats(cl_event event, int , void* data_) {
    PrintStatsData* data = (PrintStatsData*)data_;
    cl::Kernel* kernel = data->kernel;
    cl_int err;
    std::string name = kernel->getInfo<CL_KERNEL_FUNCTION_NAME>(&err);
    checkErr(err, "failed to get kernel name");

    cl_ulong queue_time_ns;
    cl_ulong end_time_ns;
    cl_ulong start_time_ns;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queue_time_ns, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time_ns, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time_ns, NULL);

    cl_ulong queued = (start_time_ns - queue_time_ns);

    cl_ulong elapsed = (end_time_ns - start_time_ns);
    double seconds = elapsed / 1.0e9;
    std::cout << name << ": elapsed = " << elapsed << "ns" << " = " << seconds << "s"
              << " = " << data->items / seconds << " op/s" << ", queued for " << queued / 1.0e9 << "s" << std::endl;

    delete data;
}
