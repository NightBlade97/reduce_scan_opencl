#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_reduce(int n,  OpenCL& opencl) {
    auto a = random_vector<float>(n);
    volatile float result = 0, expected_result = 0;

    

    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "reduce");

    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();
    cl::Buffer d_data(opencl.queue, begin(a), end(a), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, a.size()*sizeof(float));
    auto t2 = clock_type::now();

    int group_size = 64;
    bool evenOp = true;

    for(int local_size  = n; local_size > 1; local_size = (local_size + group_size - 1) / group_size) {
        
        if (evenOp) {
            kernel.setArg(0, d_data);
            kernel.setArg(1, d_result);
        }
        else{
            kernel.setArg(0, d_result);
            kernel.setArg(1, d_data);
        }

        evenOp = !evenOp;

        kernel.setArg(2, cl::Local(group_size*sizeof(float)));
        kernel.setArg(3, local_size);
        kernel.setArg(4, group_size);
        int threads_count = ((local_size + group_size - 1) / group_size) * group_size;
        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(threads_count), cl::NDRange(group_size));

    }

    auto t3 = clock_type::now();
    if (!evenOp) {
        opencl.queue.enqueueReadBuffer(d_result, true, 0, sizeof(float), begin(a));
    } 
    else {
        opencl.queue.enqueueReadBuffer(d_data, true, 0, sizeof(float), begin(a));
    }
    opencl.queue.flush();

    auto t4 = clock_type::now();
    
    result = a[0];
    // TODO Implement OpenCL version! See profile_vector_times_vector for an example.
    // TODO Uncomment the following line!
    verify_value(expected_result, result, 1e4f);
    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
}

void profile_scan_inclusive(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);

    std::vector<cl::Buffer> cl_buffers;
    std::vector<int> buffers_sizes;

    cl::Kernel kernel_main(opencl.program, "scan_inclusive");
    cl::Kernel kernel_end(opencl.program, "scan_end");

    auto t0 = clock_type::now();
    scan_inclusive(expected_result);
    auto t1 = clock_type::now();

    cl_buffers.emplace_back(opencl.queue, begin(a), end(a), true);

    auto t2 = clock_type::now();

    int group_size = 64;
    int iterator = 0;
    
    for(int local_size  = n; local_size > 1; local_size = (local_size + group_size - 1) / group_size) {
        
        kernel_main.setArg(0,cl_buffers[iterator]);

        buffers_sizes.push_back(local_size);

        cl_buffers.emplace_back(opencl.context, CL_MEM_READ_WRITE, (local_size + group_size)*sizeof(float));
        
        kernel_main.setArg(1,cl_buffers[iterator + 1]);
        
        ++iterator;

        kernel_main.setArg(2, cl::Local(group_size*sizeof(float)));
        kernel_main.setArg(3, local_size);
        kernel_main.setArg(4, group_size);
        int threads_count = ((local_size + group_size - 1) / group_size) * group_size;

        opencl.queue.enqueueNDRangeKernel(kernel_main, cl::NullRange, cl::NDRange(threads_count), cl::NDRange(group_size));
        
        //printf("Local size =  %i \n", local_size);
        opencl.queue.flush();
    }

    for(int i = iterator - 1; i >=1; --i) {
        kernel_end.setArg(0, cl_buffers[i-1]);

        kernel_end.setArg(1, cl_buffers[i]);

        kernel_end.setArg(2, buffers_sizes[i-1]);

        kernel_end.setArg(3, group_size);

        int threads_count = ((buffers_sizes[i-1] + group_size - 1) / group_size) * group_size;
        opencl.queue.enqueueNDRangeKernel(kernel_end, cl::NullRange, cl::NDRange(threads_count), cl::NDRange(group_size));
         
    }   

    auto t3 = clock_type::now();
    opencl.queue.enqueueReadBuffer(cl_buffers[0], true, 0, result.size()*sizeof(float), begin(result));
    opencl.queue.flush();

    auto t4 = clock_type::now();
    // TODO Implement OpenCL version! See profile_vector_times_vector for an example.
    // TODO Uncomment the following line!

    
    verify_vector(expected_result, result, 1e4f);
    print("scan-inclusive",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_inclusive(1024*1024*10, opencl);
}

const std::string src = R"(
kernel void reduce(global float* data, global float* result, local float* temp, int local_size, int group_size) {
    // TODO: Implement OpenCL version.

    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    int group_id = get_group_id(0);

    if (global_id < local_size){
        temp[local_id] = data[global_id];
    }
    else{
        temp[local_id] = 0.f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int block_size = group_size;
    int half_block_size = block_size / 2;

    while (half_block_size  > 0) {
        if (local_id + half_block_size < block_size) {
            temp[local_id] += temp[local_id + half_block_size];

        }
       
        block_size = half_block_size;
        half_block_size = block_size / 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (local_id == 0) {
        result[group_id] = temp[local_id];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void scan_inclusive(global float* data, global float* result, local float* temp, int local_size, int group_size) {
    // TODO: Implement OpenCL version.\

    int local_id = get_local_id(0);
    int global_id = get_global_id(0); 
    
    int group_id = get_group_id(0);

    if (global_id < local_size){
        temp[local_id] = data[global_id ];
    }
    else{
        temp[local_id] = 0.f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int offset = 1; offset < local_size; offset *= 2) {
        if (local_id >= offset && global_id < local_size) {
            temp[local_id] += temp[local_id - offset];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (global_id < local_size){
        data[global_id] = temp[local_id];
    }

    if (local_id == 0) {
        result[group_id] = temp[group_size - 1];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
}

kernel void scan_end(global float* data, global float* result, int local_size, int group_size){

    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    if (global_id >= group_size  && global_id < local_size){
        data[global_id] += result[group_id - 1];
    }

}

)";

int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
