#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <unordered_set>

#include "common.h"

const int DATA_CACHE_BLOCK_SIZE = 128;

int main() {
    const char *trace_path = getenv("TRACE_PATH");
    if (!trace_path) {
        std::cerr << "error: environment variable TRACE_PATH not set"
                  << std::endl;
        return 1;
    }

    std::ifstream trace_file(trace_path, std::ios_base::binary);
    if (!trace_file) {
        std::cerr << "error: could not open file " << trace_path << std::endl;
        return 1;
    }

    mem_access_t ma;

    double total_divergence = 0.0;
    uint64_t n_references = 0;

    while (
        trace_file.read(reinterpret_cast<char *>(&ma), sizeof(mem_access_t)) ||
        trace_file.gcount() > 0) {
        std::unordered_set<uint64_t> cache_block_addrs;
        int active_threads = 0;

        for (int i = 0; i < 32; i++) {
            if (ma.addrs[i]) {
                cache_block_addrs.insert(ma.addrs[i] / DATA_CACHE_BLOCK_SIZE);
                active_threads++;
            }
        }

        if (active_threads) {
            double divergence =
                static_cast<double>(cache_block_addrs.size()) / active_threads;
            total_divergence += divergence;
            n_references++;
        }
    }

    trace_file.close();

    if (n_references) {
        std::cout << "memory divergence = " << total_divergence / n_references
                  << std::endl;
    } else {
        std::cout << "no memory references" << std::endl;
    }
};