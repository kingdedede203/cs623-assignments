/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
// #define USE_ASYNC_STREAM
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)

enum class RecvThreadState {
    INIT,
    WORKING,
    STOP,
    FINISHED,
};

struct CTXstate {
    ChannelDev* channel_dev = nullptr;
    ChannelHost channel_host;
    CUmodule tool_module;
    CUfunction flush_channel_func;
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    bool need_sync = false;
};

#include "tool_func/flush_channel.c"

/* lock */
pthread_mutex_t mutex;
pthread_mutex_t cuda_event_mutex;

std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

void* recv_thread_fun(void* args);

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

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    /* set mutex as recursive */
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

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
        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        if (verbose) {
            printf("Inspecting function %s at address 0x%lx\n",
                   nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
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

            if (sass_to_id_map.find(instr->getSass()) == sass_to_id_map.end()) {
                int opcode_id = sass_to_id_map.size();
                sass_to_id_map[instr->getSass()] = opcode_id;
                id_to_sass_map[opcode_id] = std::string(instr->getSass());
            }

            int opcode_id = sass_to_id_map[instr->getSass()];
            std::vector<int> reg_num_list;
            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG) {
                    for (int reg_idx = 0; reg_idx < instr->getSize() / 4;
                         reg_idx++) {
                        reg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
                }
            }
            /* insert call to the instrumentation function with its
             * arguments */
            nvbit_insert_call(instr, "record_reg_val", IPOINT_BEFORE);
            /* guard predicate value */
            nvbit_add_call_arg_guard_pred_val(instr);
            /* opcode id */
            nvbit_add_call_arg_const_val32(instr, opcode_id);
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)ctx_state->channel_dev);
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
            for (int num : reg_num_list) {
                /* last parameter tells it is a variadic parameter passed to
                 * the instrument function record_reg_val() */
                nvbit_add_call_arg_reg_val(instr, num, true);
            }
            cnt++;
        }
    }
}

void init_context_state(CUcontext ctx) {
    CTXstate* ctx_state = ctx_state_map[ctx];
    ctx_state->recv_thread_done = RecvThreadState::WORKING;
    CUDA_SAFECALL(
        cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev)));
    ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
                                 ctx_state->channel_dev, recv_thread_fun, ctx);
    nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
}

static void leave_kernel_launch(CUcontext ctx, CTXstate* ctx_state) {
#ifdef USE_ASYNC_STREAM
    /* make sure current kernel is completed */
    CUDA_SAFECALL(cudaDeviceSynchronize());
    /* issue flush of channel so we are sure all the memory accesses
     * have been pushed */
    void* args[] = {&ctx_state->channel_dev};
    nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1, 1, 0,
                        nullptr, args, nullptr);
    CUDA_SAFECALL(cudaDeviceSynchronize());
#endif
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* name, void* params, CUresult* pStatus) {
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
    CTXstate* ctx_state = ctx_state_map[ctx];

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* Make sure GPU is idle */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

            ctx_state->need_sync = true;
            instrument_function_if_needed(ctx, func);

            nvbit_enable_instrumented(ctx, func, true);

            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                printf(
                    "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - "
                    "nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    nvbit_get_func_name(ctx, func), p->config->gridDimX,
                    p->config->gridDimY, p->config->gridDimZ,
                    p->config->blockDimX, p->config->blockDimY,
                    p->config->blockDimZ, nregs,
                    shmem_static_nbytes + p->config->sharedMemBytes,
                    (uint64_t)p->config->hStream);
            } else {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                printf(
                    "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - "
                    "nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    nvbit_get_func_name(ctx, func), p->gridDimX, p->gridDimY,
                    p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ,
                    nregs, shmem_static_nbytes + p->sharedMemBytes,
                    (uint64_t)p->hStream);
            }
        } else {
            leave_kernel_launch(ctx, ctx_state);
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;

    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);
    char* recv_buffer = (char*)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

        if (num_recv_bytes > 0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                reg_info_t* ri = (reg_info_t*)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ri->cta_id_x == -1) {
                    break;
                }

                printf("CTA %d,%d,%d - warp %d - %s:\n", ri->cta_id_x,
                       ri->cta_id_y, ri->cta_id_z, ri->warp_id,
                       id_to_sass_map[ri->opcode_id].c_str());

                for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
                    printf("* ");
                    for (int i = 0; i < 32; i++) {
                        printf("Reg%d_T%d: 0x%08x ", reg_idx, i,
                               ri->reg_vals[i][reg_idx]);
                    }
                    printf("\n");
                }

                printf("\n");
                num_processed_bytes += sizeof(reg_info_t);
            }
        }
    }
    free(recv_buffer);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    assert(ctx_state_map.find(ctx) == ctx_state_map.end());
    CTXstate* ctx_state = new CTXstate;
    ctx_state_map[ctx] = ctx_state;

    nvbit_load_tool_module(ctx, (const void*)flush_channel_bin,
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
    assert(ctx_state_map.find(ctx) != ctx_state_map.end());
    CTXstate* ctx_state = ctx_state_map[ctx];

    if (ctx_state->need_sync) {
        void* args[] = {&ctx_state->channel_dev};
        nvbit_launch_kernel(ctx, ctx_state->flush_channel_func, 1, 1, 1, 1, 1,
                            1, 0, nullptr, args, nullptr);
        CUDA_SAFECALL(cudaDeviceSynchronize());
    }

    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    if (ctx_state->recv_thread_done != RecvThreadState::INIT) {
        ctx_state->recv_thread_done = RecvThreadState::STOP;
        while (ctx_state->recv_thread_done != RecvThreadState::FINISHED);
    }

    ctx_state->channel_host.destroy(false);
    cudaFree(ctx_state->channel_dev);
    skip_callback_flag = false;
    delete ctx_state;
    ctx_state_map.erase(ctx);
    pthread_mutex_unlock(&mutex);
}
