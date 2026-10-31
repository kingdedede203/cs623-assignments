#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using std::cerr;
using std::cout;
using std::fabs;
using std::floor;
using std::sort;
using std::vector;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)                                                    \
    {                                                                          \
        gpuAssert((ans), __FILE__, __LINE__);                                  \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        cerr << "GPUassert: " << cudaGetErrorString(code) << ' ' << file << ' '
             << line << '\n';
        if (abort)
            exit(code);
    }
}

constexpr uint32_t Num_Variables = 10;
constexpr int Last = Num_Variables - 1;

__constant__ uint32_t M[Num_Variables];
__constant__ double START[Num_Variables], STEP[Num_Variables];
__constant__ double C[Num_Variables * Num_Variables], D[Num_Variables],
    E[Num_Variables];

// compile with -fmad=false
__global__ void __launch_bounds__(256)
    grid_search(uint64_t segment_start, uint32_t segment_size, uint64_t total,
                unsigned long long *d_n_pnts, uint64_t *d_pnts,
                uint32_t capacity) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t start = segment_start + static_cast<uint64_t>(tid) * segment_size;

    if (start >= total)
        return;

    uint32_t n = static_cast<uint32_t>(
        min(total - start, static_cast<uint64_t>(segment_size)));

    // do this only once; modulus and division segment_size times a thread is
    // too expensive
    uint32_t r[Num_Variables];
    uint64_t tmp = start;
#pragma unroll
    for (int i = Last; i >= 0; i--) {
        r[i] = tmp % M[i];
        tmp /= M[i];
    }

    double p[Num_Variables], s;

    for (uint32_t k = 0; k < n;) {
#pragma unroll
        for (int i = 0; i < Num_Variables; i++) {
            s = 0.0;

#pragma unroll
            for (int j = 0; j < Last; j++)
                s += C[i * Num_Variables + j] * (START[j] + r[j] * STEP[j]);

            p[i] = s; // p[i] = Ci1 x1 + Ci2 x2 + ... + Ci9 x9
        }

        // innermost loop over x10
        for (; r[Last] < M[Last] && k < n; r[Last]++, k++) {
            double x = START[Last] + r[Last] * STEP[Last];
            bool valid = true;

#pragma unroll
            for (int i = 0; i < Num_Variables; i++) {
                if (fabs(p[i] + C[i * Num_Variables + Last] * x - D[i]) >
                    E[i]) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                uint64_t idx = atomicAdd(d_n_pnts, 1ULL);
                if (idx < capacity)
                    d_pnts[idx] = start + k;
            }
        }

        // increment coordinate
        r[Last] = 0;
#pragma unroll
        for (int i = Last - 1; i >= 0; i--) {
            if (++r[i] < M[i]) // no carry - break
                break;

            r[i] = 0;
        }
    }
}

int main() {
    auto *h_a = new double[Num_Variables *
                           (Num_Variables + 2)]; // per entry: coefficients,
                                                 // RHS, error; row major
    auto *h_b =
        new double[Num_Variables *
                   3]; // per entry: grid start, grid stop, grid step; row major

    std::ifstream disp_file("disp.txt");
    for (int i = 0; i < Num_Variables * (Num_Variables + 2); i++) {
        disp_file >> h_a[i];
    }
    disp_file.close();

    std::ifstream grid_file("grid.txt");
    for (int i = 0; i < Num_Variables * 3; i++) {
        grid_file >> h_b[i];
    }
    grid_file.close();

    // get device properties
    int device;
    cudaCheckError(cudaGetDevice(&device));
    cudaDeviceProp prop;
    cudaCheckError(cudaGetDeviceProperties(&prop, device));
    cout << "Maximum threads per block = " << prop.maxThreadsPerBlock << '\n'
         << "Maximum threads per SM = " << prop.maxThreadsPerMultiProcessor
         << '\n'
         << "Maximum (32-bit) registers per SM = " << prop.regsPerMultiprocessor
         << '\n'
         << "Maximum blocks per SM = " << prop.maxBlocksPerMultiProcessor
         << '\n'
         << "Number of SMs = " << prop.multiProcessorCount << '\n';

    cudaCheckError(cudaFree(0));

    HRTimer start = HR::now();

    double *h_START = new double[Num_Variables],
           *h_STEP = new double[Num_Variables];
    uint32_t *h_M = new uint32_t[Num_Variables];

    for (int i = 0; i < Num_Variables; i++) {
        h_START[i] = h_b[i * 3];
        h_STEP[i] = h_b[i * 3 + 2];
        h_M[i] = static_cast<uint32_t>(
            floor((h_b[i * 3 + 1] - h_b[i * 3]) / h_b[i * 3 + 2]));
    }

    double *h_C = new double[Num_Variables * Num_Variables],
           *h_D = new double[Num_Variables], *h_E = new double[Num_Variables],
           kk = 0.3;

    for (int i = 0; i < Num_Variables; i++) {
        for (int j = 0; j < Num_Variables; j++)
            h_C[i * Num_Variables + j] = h_a[i * (Num_Variables + 2) + j];

        h_D[i] = h_a[i * (Num_Variables + 2) + Num_Variables];
        h_E[i] = kk * h_a[i * (Num_Variables + 2) + Num_Variables + 1];
    }

    cudaCheckError(
        cudaMemcpyToSymbol(M, h_M, Num_Variables * sizeof(uint32_t)));
    cudaCheckError(
        cudaMemcpyToSymbol(START, h_START, Num_Variables * sizeof(double)));
    cudaCheckError(
        cudaMemcpyToSymbol(STEP, h_STEP, Num_Variables * sizeof(double)));
    cudaCheckError(cudaMemcpyToSymbol(
        C, h_C, Num_Variables * Num_Variables * sizeof(double)));
    cudaCheckError(cudaMemcpyToSymbol(D, h_D, Num_Variables * sizeof(double)));
    cudaCheckError(cudaMemcpyToSymbol(E, h_E, Num_Variables * sizeof(double)));

    uint32_t threads_per_block = 256;
    uint32_t blocks_per_grid = 4096;
    uint32_t segment_size = 1024;

    uint64_t pnts_per_launch = static_cast<uint64_t>(threads_per_block) *
                               blocks_per_grid * segment_size;
    uint32_t capacity = 1 << 20;

    uint64_t total = 1;
    for (int i = 0; i < Num_Variables; i++)
        total *= h_M[i];

    vector<uint64_t> h_pnts;
    h_pnts.resize(capacity);

    unsigned long long *d_n_pnts, h_n_pnts;
    uint64_t *d_pnts;

    cudaCheckError(cudaMalloc(&d_n_pnts, sizeof(unsigned long long)));
    cudaCheckError(cudaMalloc(&d_pnts, capacity * sizeof(uint64_t)));

    uint64_t segment_start = 0, total_pnts = 0;

    std::ofstream results_file("results-v1.txt");
    if (!results_file) {
        cerr << "error creating file\n";
        return 1;
    }
    results_file << std::fixed << std::setprecision(6);

    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    float kernel_ms;
    double total_kernel_ms = 0.0;

    while (segment_start < total) {
        cudaCheckError(cudaMemset(d_n_pnts, 0, sizeof(unsigned long long)));

        cudaEventRecord(kernel_start);

        grid_search<<<blocks_per_grid, threads_per_block>>>(
            segment_start, segment_size, total, d_n_pnts, d_pnts, capacity);
        cudaCheckError(cudaGetLastError());

        cudaEventRecord(kernel_end);
        cudaEventSynchronize(kernel_end);

        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end);

        total_kernel_ms += kernel_ms;

        cudaCheckError(cudaMemcpy(&h_n_pnts, d_n_pnts,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));

        if (h_n_pnts > capacity) {
            capacity = h_n_pnts;

            cudaCheckError(cudaFree(d_pnts));
            cudaCheckError(cudaMalloc(&d_pnts, capacity * sizeof(uint64_t)));

            h_pnts.resize(capacity);

            continue;
        }

        cudaCheckError(cudaMemcpy(h_pnts.data(), d_pnts,
                                  h_n_pnts * sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost));

        // launches cover increasing id ranges, so sorting each batch gives the
        // same global order as the C loop nest
        sort(h_pnts.begin(), h_pnts.begin() + h_n_pnts);

        for (uint64_t k = 0; k < h_n_pnts; k++) {
            uint64_t tmp = h_pnts[k];
            uint32_t r[Num_Variables];
            for (int i = Last; i >= 0; i--) {
                r[i] = tmp % h_M[i];
                tmp /= h_M[i];
            }

            for (int i = 0; i < Last; i++)
                results_file << h_START[i] + r[i] * h_STEP[i] << '\t';
            results_file << h_START[Last] + r[Last] * h_STEP[Last] << '\n';
        }

        total_pnts += h_n_pnts;

        segment_start += pnts_per_launch;
    }

    results_file.close();

    HRTimer end = HR::now();

    cout << "Total kernel time = " << total_kernel_ms / 1000.0 << " s\n";
    cout << "End-to-end time = " << duration<double>(end - start).count()
         << " s\n";

    cout << "Total points = " << total_pnts << '\n';

    delete[] h_a;
    delete[] h_b;
    delete[] h_M;
    delete[] h_START;
    delete[] h_STEP;
    delete[] h_C;
    delete[] h_D;
    delete[] h_E;

    cudaCheckError(cudaFree(d_n_pnts));
    cudaCheckError(cudaFree(d_pnts));
}