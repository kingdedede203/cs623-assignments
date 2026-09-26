#include <cstdint>
#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <vector>

#include "common.h"

struct Node {
    Node *prev;
    Node *next;
    uint64_t tag;
};

class Cache {
  public:
    uint32_t block_offset_bits, set_index_bits, n_sets, n_ways;
    std::vector<std::vector<uint64_t>> sets;

    Cache(uint32_t block_size, uint32_t n_sets, uint32_t n_ways)
        : n_sets(n_sets), n_ways(n_ways) {
        block_offset_bits = __builtin_ctz(block_size);
        set_index_bits = __builtin_ctz(n_sets);
        sets.resize(n_sets);
        for (auto &set : sets) {
            set.reserve(n_ways);
        }
    }

    bool fetch(uint64_t addr) {
        uint64_t block_addr = addr >> block_offset_bits;
        uint64_t set_idx = block_addr & (n_sets - 1);
        uint64_t tag = block_addr >> set_index_bits;

        auto &current_set = sets[set_idx];

        for (auto it = current_set.begin(); it != current_set.end(); ++it) {
            if (*it == tag) {
                current_set.erase(it);
                current_set.push_back(tag);
                return true;
            }
        }

        if (current_set.size() == n_ways) {
            current_set.erase(current_set.begin());
        }
        current_set.push_back(tag);
        return false;
    }
};

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

    std::vector<std::vector<std::unique_ptr<Cache>>> all_caches;
    for (int n = 1; n <= 64; n *= 2) {
        std::vector<std::unique_ptr<Cache>> sm_caches;
        for (int i = 0; i < n; i++) {
            sm_caches.push_back(std::make_unique<Cache>(128, 32, 32));
        }
        all_caches.push_back(std::move(sm_caches));
    }

    std::vector<int> hits(7, 0), misses(7, 0);
    mem_access_t ma;

    while (
        trace_file.read(reinterpret_cast<char *>(&ma), sizeof(mem_access_t))) {
        for (uint64_t addr : ma.addrs) {
            if (addr != 0) {
                int config_idx = 0;
                for (int n = 1; n <= 64; n *= 2, config_idx++) {
                    if (all_caches[config_idx][ma.cta_id_x % n]->fetch(addr)) {
                        hits[config_idx]++;
                    } else {
                        misses[config_idx]++;
                    }
                }
            }
        }
    }

    for (int i = 0; i < 7; i++) {
        std::cout << hits[i] << ' ' << misses[i] << '\n';
    }
}