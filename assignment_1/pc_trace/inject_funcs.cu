#include <stdint.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

extern "C" __device__ __noinline__ void instrument_mem(uint64_t pc,
                                                       uint64_t pchannel_dev) {

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (laneid != first_laneid) {
        return;
    }

    /* first active lane pushes information on the channel */
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&pc, sizeof(uint64_t));
}
