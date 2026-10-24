#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)                                                    \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

// Evaluate with three sizes, 64, 128, and 256.
#ifndef INPUT_WIDTH
#define INPUT_WIDTH 64
#endif

const uint32_t Input_Width = INPUT_WIDTH;
const uint32_t Input_Num_Elems = (Input_Width * Input_Width * Input_Width);
const uint32_t Input_Size_Bytes = Input_Num_Elems * sizeof(float);

const uint32_t Filter_Width = 5;
const uint32_t Filter_Num_Elems = (Filter_Width * Filter_Width * Filter_Width);
const uint32_t Filter_Size_Bytes = Filter_Num_Elems * sizeof(float);

// User Definitions
const uint32_t Block_Width = 8;
const uint32_t Block_Num_Elems = Block_Width * Block_Width * Block_Width;
const uint32_t Tile_Width = Block_Width + Filter_Width - 1;
const uint32_t Tile_Num_Elems = Tile_Width * Tile_Width * Tile_Width;
const int32_t Filter_Width_L = Filter_Width / 2;
const int32_t Filter_Width_R = (Filter_Width + 1) / 2;
const uint32_t Runs = 10;

// Implement a single-threaded CPU version
void seq_cpu_conv(const float *h_input, const float *h_filter,
                  float *h_output) {
    // flattened = z * K * K + y * K + x
    for (int i = 0; i < Input_Num_Elems; i++) {
        int input_x = i % Input_Width,
            input_y = (i / Input_Width) % Input_Width,
            input_z = i / (Input_Width * Input_Width);

        float output = 0.0;

        for (int z = max(0, input_z - Filter_Width_L);
             z < min(input_z + Filter_Width_R, Input_Width); z++) {
            for (int y = max(0, input_y - Filter_Width_L);
                 y < min(input_y + Filter_Width_R, Input_Width); y++) {
                for (int x = max(0, input_x - Filter_Width_L);
                     x < min(input_x + Filter_Width_R, Input_Width); x++) {
                    int filter_x = x - input_x + Filter_Width_L,
                        filter_y = y - input_y + Filter_Width_L,
                        filter_z = z - input_z + Filter_Width_L;
                    int input_idx =
                        z * Input_Width * Input_Width + y * Input_Width + x;
                    int filter_idx = filter_z * Filter_Width * Filter_Width +
                                     filter_y * Filter_Width + filter_x;
                    output += h_input[input_idx] * h_filter[filter_idx];
                }
            }
        }

        h_output[i] = output;
    }
}

// Implement a naive version using loads and stores to global memory. Edit
// the function definition as required.
__global__ void basic_kernel(const float *d_input, const float *d_filter,
                             float *d_output) {
    int input_x = blockIdx.x * blockDim.x + threadIdx.x,
        input_y = blockIdx.y * blockDim.y + threadIdx.y,
        input_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (input_x >= Input_Width || input_y >= Input_Width ||
        input_z >= Input_Width) {
        return;
    }

    float output = 0.0;

    for (int z = max(0, input_z - Filter_Width_L);
         z < min(input_z + Filter_Width_R, Input_Width); z++) {
        for (int y = max(0, input_y - Filter_Width_L);
             y < min(input_y + Filter_Width_R, Input_Width); y++) {
            for (int x = max(0, input_x - Filter_Width_L);
                 x < min(input_x + Filter_Width_R, Input_Width); x++) {
                int filter_x = x - input_x + Filter_Width_L,
                    filter_y = y - input_y + Filter_Width_L,
                    filter_z = z - input_z + Filter_Width_L;
                int input_idx =
                    z * Input_Width * Input_Width + y * Input_Width + x;
                int filter_idx = filter_z * Filter_Width * Filter_Width +
                                 filter_y * Filter_Width + filter_x;
                output += d_input[input_idx] * d_filter[filter_idx];
            }
        }
    }

    d_output[input_z * Input_Width * Input_Width + input_y * Input_Width +
             input_x] = output;
}

// Use constant memory for accessing the filter and shared memory for
// storing the input array including halo cells.
__constant__ float dc_filter[Filter_Num_Elems];

__global__ void opt_kernel(const float *d_input, float *d_output) {
    __shared__ float tile[Tile_Num_Elems];

    int thread_idx = threadIdx.z * Block_Width * Block_Width +
                     threadIdx.y * Block_Width + threadIdx.x;

    for (int i = thread_idx; i < Tile_Num_Elems; i += Block_Num_Elems) {
        int tile_x = i % Tile_Width, tile_y = (i / Tile_Width) % Tile_Width,
            tile_z = i / (Tile_Width * Tile_Width);

        int input_x = blockIdx.x * blockDim.x + tile_x - Filter_Width_L,
            input_y = blockIdx.y * blockDim.y + tile_y - Filter_Width_L,
            input_z = blockIdx.z * blockDim.z + tile_z - Filter_Width_L;

        if (input_x < 0 || input_x >= Input_Width || input_y < 0 ||
            input_y >= Input_Width || input_z < 0 || input_z >= Input_Width) {
            tile[i] = 0.0f;
            continue;
        }

        int input_idx = input_z * Input_Width * Input_Width +
                        input_y * Input_Width + input_x;

        tile[i] = d_input[input_idx];
    }

    __syncthreads();

    float output = 0.0f;

#pragma unroll
    for (int i = 0; i < Filter_Num_Elems; i++) {
        int filter_x = i % Filter_Width,
            filter_y = (i / Filter_Width) % Filter_Width,
            filter_z = i / (Filter_Width * Filter_Width);
        output += tile[(threadIdx.z + filter_z) * Tile_Width * Tile_Width +
                       (threadIdx.y + filter_y) * Tile_Width + threadIdx.x +
                       filter_x] *
                  dc_filter[i];
    }

    d_output[(blockIdx.z * blockDim.z + threadIdx.z) * Input_Width *
                 Input_Width +
             (blockIdx.y * blockDim.y + threadIdx.y) * Input_Width +
             blockIdx.x * blockDim.x + threadIdx.x] = output;
}

void check_result(const float *w_ref, const float *w_opt, const uint32_t N) {
    double maxdiff = 0.0;
    int numdiffs = 0;

    for (uint64_t i = 0; i < (N * N * N); i++) {
        double this_diff = w_ref[i] - w_opt[i];
        if (std::fabs(this_diff) > THRESHOLD) {
            numdiffs++;
            if (this_diff > maxdiff) {
                maxdiff = this_diff;
            }
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
             << "; Max Diff = " << maxdiff << endl;
    } else {
        cout << "No differences found between base and test versions\n";
    }
}

void print3D(const float *A, const uint32_t N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                cout << A[i * N * N + j * N + k] << "\t";
            }
            cout << "\n";
        }
        cout << "\n";
    }
}

int main() {
    auto *h_input = new float[Input_Num_Elems];
    std::fill_n(h_input, Input_Num_Elems, 1.0F);

    const float h_filter3D[Filter_Width][Filter_Width][Filter_Width] = {
        {2.0F, 3.0F, 4.0F, 3.0F, 2.0F},
        {4.0F, 5.0F, 6.0F, 5.0F, 4.0F},
        {3.0F, 4.0F, 5.0F, 4.0F, 3.0F}};
    auto *h_filter = new float[Filter_Num_Elems];
    memcpy(h_filter, h_filter3D, Filter_Size_Bytes);

    auto *h_output_cpu = new float[Input_Num_Elems];
    HRTimer start = HR::now();
    seq_cpu_conv(h_input, h_filter, h_output_cpu);
    HRTimer end = HR::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Serial version: " << duration << " ms\n";

    float *d_input = nullptr, *d_filter = nullptr, *d_output = nullptr;

    float total_kernel_time = 0.0f;
    cudaEvent_t kernel_start, kernel_stop;
    cudaCheckError(cudaEventCreate(&kernel_start));
    cudaCheckError(cudaEventCreate(&kernel_stop));

    // Invoke kernel3D_basic()
    cudaCheckError(cudaMalloc(&d_input, Input_Size_Bytes));
    cudaCheckError(cudaMalloc(&d_filter, Filter_Size_Bytes));
    cudaCheckError(cudaMalloc(&d_output, Input_Size_Bytes));
    cudaCheckError(
        cudaMemcpy(d_input, h_input, Input_Size_Bytes, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_filter, h_filter, Filter_Size_Bytes,
                              cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(Block_Width, Block_Width, Block_Width);
    // Input_Width is guaranteed to be a multiple of Block_Width = 8
    dim3 blocksPerGrid(Input_Width / Block_Width, Input_Width / Block_Width,
                       Input_Width / Block_Width);

    // TODO: calculate end-to-end time
    // TODO: untimed kernel launch to account for one time start up costs
    cudaCheckError(cudaEventRecord(kernel_start));
    for (int i = 0; i < Runs; i++) {
        basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_filter,
                                                         d_output);
    }
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(kernel_stop));
    cudaCheckError(cudaEventSynchronize(kernel_stop));

    cudaCheckError(
        cudaEventElapsedTime(&total_kernel_time, kernel_start, kernel_stop));

    // Adapt check_result()
    auto *h_output_gpu = new float[Input_Num_Elems];
    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, Input_Size_Bytes,
                              cudaMemcpyDeviceToHost));
    check_result(h_output_cpu, h_output_gpu, Input_Width);

    cout << "Kernel3D_basic time (ms): " << total_kernel_time / Runs << "\n";

    cudaCheckError(cudaMemset(d_output, 0, Input_Size_Bytes));

    // Invoke kernel3D_opt()
    cudaCheckError(cudaMemcpyToSymbol(dc_filter, h_filter, Filter_Size_Bytes));

    cudaCheckError(cudaEventRecord(kernel_start));
    for (int i = 0; i < Runs; i++) {
        opt_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    }
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaEventRecord(kernel_stop));
    cudaCheckError(cudaEventSynchronize(kernel_stop));

    cudaCheckError(
        cudaEventElapsedTime(&total_kernel_time, kernel_start, kernel_stop));

    // Adapt check_result()
    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, Input_Size_Bytes,
                              cudaMemcpyDeviceToHost));
    check_result(h_output_cpu, h_output_gpu, Input_Width);
    cout << "Kernel3D_opt time (ms): " << total_kernel_time / Runs << "\n";

    // Free memory
    cudaCheckError(cudaEventDestroy(kernel_start));
    cudaCheckError(cudaEventDestroy(kernel_stop));

    cudaCheckError(cudaFree(d_input));
    cudaCheckError(cudaFree(d_filter));
    cudaCheckError(cudaFree(d_output));

    delete[] h_input;
    delete[] h_filter;
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    return EXIT_SUCCESS;
}
