#include <assert.h>
#include <cstdint>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

// #define USE_ASYNC_STREAM
// NOTE: USE_ASYNC_STREAM version of channel.hpp can cause deadlock if two
// kernels have some dependency, since after each kernel launch, a device
// synchronization is needed for USE_ASYNC_STREAM version. Otherwise, the async
// copy in ChannelHost->recv() and a CUDA API call following a kernel launch
// can cause a deadlock, where the CUDA API call can take the CUDA context lock
// and waits for the kernel to finish, but the async copy in ChannelHost->recv()
// is waiting for the aforementioned CUDA context lock, thus stalling the kernel
// to finish (the instrumented kernel now needs to flush channel bufferto host
// side to finish by using the async copy).

/* for channel */
#include "utils/channel.hpp"

#define HEX(x)                                                                 \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x      \
         << std::dec

#define CHANNEL_SIZE (1l << 20)

enum class RecvThreadState {
    INIT,
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    /* context id */
    int id;

    /* Channel used to communicate from GPU to CPU receiving thread */
    ChannelDev *channel_dev = nullptr;
    ChannelHost channel_host;

    /* tool module */
    CUmodule tool_module;

    /* flush channel function */
    CUfunction flush_channel_func;

    // Start with INIT, so that if no kernel is launched in the ctx, there is
    // no need to wait on the thread at the context termination.
    // After initialization, set it to WORKING to make recv thread get data,
    // parent thread sets it to STOP to make recv thread stop working.
    // recv thread sets it to FINISHED when it cleans up.
    // parent thread should wait until the state becomes FINISHED to clean up.
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    // whether the context and the channel need a synchronization.
    bool need_sync = false;
};

#include "tool_func/flush_channel.c"

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;
pthread_mutex_t instr_addr_mutex;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate *> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

std::set<uint64_t> instr_addrs;

/* grid launch id, incremented at every launch */
uint64_t global_grid_launch_id = 0;

void *recv_thread_fun(void *args);

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());

    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);
    pthread_mutex_init(&instr_addr_mutex, nullptr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate *ctx_state = ctx_state_map[ctx];

    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* get vector of instructions of function "f" */
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        uint64_t func_addr = nvbit_get_func_addr(ctx, f);

        if (verbose) {
            printf(
                "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
                "0x%lx\n",
                ctx, f, nvbit_get_func_name(ctx, f), func_addr);
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
                cnt++;
                continue;
            }
            if (verbose) {
                instr->printDecoded();
            }

            nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
            nvbit_add_call_arg_const_val64(instr,
                                           func_addr + instr->getOffset());
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)ctx_state->channel_dev);

            cnt++;
        }
    }
}

void init_context_state(CUcontext ctx) {
    CTXstate *ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    CUDA_SAFECALL(
        cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev)));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

static void enter_kernel_launch(CUcontext ctx, CUfunction func,
                                uint64_t &grid_launch_id, nvbit_api_cuda_t cbid,
                                void *params, bool stream_capture = false,
                                bool build_graph = false) {
    /* instrument */
    instrument_function_if_needed(ctx, func);

    int nregs = 0;
    CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

    int shmem_static_nbytes = 0;
    CUDA_SAFECALL(cuFuncGetAttribute(
        &shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

    /* get function name and pc */
    const char *func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    // during stream capture or manual graph build, no kernel is launched, so
    // do not set launch argument, do not print kernel info, do not increase
    // grid_launch_id. All these should be done at graph node launch time.
    if (!stream_capture && !build_graph) {
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            printf("MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                   "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                   "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                   "id %ld\n",
                   (uint64_t)ctx, pc, func_name, grid_launch_id,
                   p->config->gridDimX, p->config->gridDimY,
                   p->config->gridDimZ, p->config->blockDimX,
                   p->config->blockDimY, p->config->blockDimZ, nregs,
                   shmem_static_nbytes + p->config->sharedMemBytes,
                   (uint64_t)p->config->hStream);
        } else {
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            printf("MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
                   "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
                   "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
                   "id %ld\n",
                   (uint64_t)ctx, pc, func_name, grid_launch_id, p->gridDimX,
                   p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY,
                   p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes,
                   (uint64_t)p->hStream);
        }

        // increment grid launch id for next launch
        // grid id can be changed here, since nvbit_set_at_launch() has copied
        // its value above.
        grid_launch_id++;
    }

    /* enable instrumented code to run */
    nvbit_enable_instrumented(ctx, func, true);
}

static void leave_kernel_launch(CUcontext ctx, CTXstate *ctx_state) {
#ifdef USE_ASYNC_STREAM
    // make sure user kernel finishes to avoid deadlock
    CUDA_SAFECALL(cudaDeviceSynchronize());
    /* push a flush channel kernel */
    void *args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0,
                        nullptr, args, nullptr);

    /* Make sure GPU is idle */
    CUDA_SAFECALL(cudaDeviceSynchronize());
#endif
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    /* Skip callbacks for contexts not yet initialized in ctx_state_map
     * (e.g. green-context-derived CUcontexts where nvbit_at_ctx_init has
     * not yet run). Using operator[] on an unknown key would insert a null
     * CTXstate* and cause a segfault when dereferenced below. */
    if (ctx_state_map.find(ctx) == ctx_state_map.end()) {
        skip_callback_flag = false;
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }

    CTXstate *ctx_state = ctx_state_map[ctx];

    switch (cbid) {
    // Identify all the possible CUDA launch events without stream
    // parameters, they will not get involved with cuda graph
    case API_CUDA_cuLaunch:       // deprecated
    case API_CUDA_cuLaunchGrid: { // deprecated
        cuLaunch_params *p = (cuLaunch_params *)params;
        CUfunction func = p->f;
        if (!is_exit) {
            ctx_state->need_sync = true;
            enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, params);
        } else {
            leave_kernel_launch(ctx, ctx_state);
        }
    } break;
    // To support kernel launched by cuda graph (in addition to existing
    // kernel launche method), we need to do:
    //
    // 1. instrument kernels at cudaGraphAddKernelNode event. This is for
    // cases that kernels are manually added to a cuda graph.
    // 2. distinguish captured kernels when kernels are recorded to a graph
    // using stream capture. cudaStreamIsCapturing() tells us whether a
    // stream is capturiong.
    // 3. per-kernel instruction counters, since cuda graph can launch
    // multiple kernels at the same time.
    //
    // Three cases:
    //
    // 1. original kernel launch:
    //     1a. for any kernel launch without using a stream, we instrument
    //     it before it is launched, call cudaDeviceSynchronize after it is
    //     launched and read the instruction counter of the kernel.
    //     1b. for any kernel launch using a stream, but the stream is not
    //     capturing, we do the same thing as 1a.
    //
    //  2. cuda graph using stream capturing: if a kernel is launched in a
    //  stream and the stream is capturing. We instrument the kernel before
    //  it is launched and do nothing after it is launched, because the
    //  kernel is not running until cudaGraphLaunch. Instead, we issue a
    //  cudaStreamSynchronize after cudaGraphLaunch is done and reset the
    //  instruction counters, since a cloned graph might be launched
    //  afterwards.
    //
    //  3. cuda graph manual: we instrument the kernel added by
    //  cudaGraphAddKernelNode and do the same thing for cudaGraphLaunch
    //  as 2.
    //
    // The above method should handle most of cuda graph launch cases.
    // kernel launches with stream parameter, they can be used for cuda
    // graph
    case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchCooperativeKernel:
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
    case API_CUDA_cuLaunchKernelEx:
    case API_CUDA_cuLaunchKernelEx_ptsz:
    case API_CUDA_cuLaunchGridAsync: {
        CUfunction func;
        CUstream hStream;

        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            func = p->f;
            hStream = p->config->hStream;
        } else if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                   cbid == API_CUDA_cuLaunchKernel ||
                   cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
                   cbid == API_CUDA_cuLaunchCooperativeKernel) {
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            func = p->f;
            hStream = p->hStream;
        } else {
            cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
            func = p->f;
            hStream = p->hStream;
        }

        cudaStreamCaptureStatus streamStatus;
        /* check if the stream is capturing, if yes, do not sync */
        CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
        if (!is_exit) {
            bool stream_capture =
                (streamStatus == cudaStreamCaptureStatusActive);
            ctx_state->need_sync = true;
            enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, params,
                                stream_capture);
        } else {
            if (streamStatus != cudaStreamCaptureStatusActive) {
                if (verbose >= 1) {
                    printf("kernel %s not captured by cuda graph\n",
                           nvbit_get_func_name(ctx, func));
                }
                leave_kernel_launch(ctx, ctx_state);
            } else {
                if (verbose >= 1) {
                    printf("kernel %s captured by cuda graph\n",
                           nvbit_get_func_name(ctx, func));
                }
            }
        }
    } break;
    case API_CUDA_cuGraphAddKernelNode: {
        cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
        CUfunction func = p->nodeParams->func;

        if (!is_exit) {
            // cuGraphAddKernelNode_params->nodeParams is the same as
            // cuLaunchKernel_params up to sharedMemBytes
            ctx_state->need_sync = true;
            enter_kernel_launch(ctx, func, global_grid_launch_id, cbid,
                                (void *)p->nodeParams, false, true);
        }
    } break;
    default:
        break;
    };

    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void *recv_thread_fun(void *args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate *ctx_state = ctx_state_map[ctx];

    ChannelHost *ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        /* receive buffer from channel */
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            pthread_mutex_lock(&instr_addr_mutex);
            while (num_processed_bytes < num_recv_bytes) {
                uint64_t *instr_addr =
                    (uint64_t *)&recv_buffer[num_processed_bytes];

                instr_addrs.insert(*instr_addr);

                num_processed_bytes += sizeof(uint64_t);
            }
            pthread_mutex_unlock(&instr_addr_mutex);
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    if (verbose) {
        printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
    }
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate *ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;

    nvbit_load_tool_module(ctx, (const void *)flush_channel_bin,
                           &ctx_state->tool_module);
    nvbit_find_function_by_name(ctx, ctx_state->tool_module, "flush_channel",
                                &ctx_state->flush_channel_func);

    pthread_mutex_unlock(&mutex);
}

void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    init_context_state(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    if (verbose) {
        printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
    }
    /* get context state from map */
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate *ctx_state = ctx_state_map[ctx];

    // flush channel if there is a kernel launch before context termination
    // without a device synchronization.
    if (ctx_state->need_sync) {
        void *args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1,
                            1, 0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED)
            ;
    }

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    ctx_state_map.erase(ctx);
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream,
                                uint64_t launch_handle) {
    func_config_t config = {0};
    const char *func_name = nvbit_get_func_name(ctx, func);
    uint64_t pc = nvbit_get_func_addr(ctx, func);

    pthread_mutex_lock(&mutex);
    nvbit_get_func_config(ctx, func, &config);

    printf("MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - "
           "Kernel name %s - grid launch id %ld - grid size %d,%d,%d "
           "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream "
           "id %ld\n",
           (uint64_t)ctx, pc, func_name, global_grid_launch_id, config.gridDimX,
           config.gridDimY, config.gridDimZ, config.blockDimX, config.blockDimY,
           config.blockDimZ, config.num_registers,
           config.shmem_static_nbytes + config.shmem_dynamic_nbytes,
           (uint64_t)stream);
    // grid id can be changed here, since nvbit_set_at_launch() has copied its
    // value above.
    global_grid_launch_id++;
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_term() {
    std::set<uint64_t> instr_cache_addrs;

    for (auto addr : instr_addrs) {
        instr_cache_addrs.insert(addr / 32);
    }

    printf("# unique instructions = %lu, # unique instruction cache blocks = "
           "%lu\n",
           instr_addrs.size(), instr_cache_addrs.size());
}
