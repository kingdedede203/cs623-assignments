/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

extern "C" __device__ __noinline__ void instrument_tma_mem(int pred, int opcode_id,
                                                       uint64_t grid_launch_id,
                                                       uint64_t tma_param_handle,
                                                       uint32_t tma_param_handle_size,
                                                       uint64_t pchannel_dev) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    tma_access_t ma;

    int4 cta = get_ctaid();
    int4 cluster_cta = get_cluster_ctaid();
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.cluster_cta_id_x = cluster_cta.x;
    ma.cluster_cta_id_y = cluster_cta.y;
    ma.cluster_cta_id_z = cluster_cta.z;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;

    /* collect tma args */
    memset(ma.tma_param_handle, 0, sizeof(ma.tma_param_handle));
    memcpy(ma.tma_param_handle, (uint8_t *) tma_param_handle, tma_param_handle_size);
    ma.tma_param_handle_ptr = tma_param_handle;
    ma.tma_param_handle_size = tma_param_handle_size;
    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ma, sizeof(tma_access_t));
    }
}