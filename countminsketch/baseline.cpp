#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <cstdint>
#include <ostream>

//hash function should take an item and a seed and return the hash value
using HashFunctionPtr = uint32_t(*)(uint32_t, uint32_t);

uint32_t MurmurHash3(uint32_t key, uint32_t seed) {
    key ^= seed;
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

class BaselineCountMinSketch {

    // K - number of buckets (columns)
    // N - number of sketches (rows)
    private:
        std::vector<std::vector<uint32_t>> estimates;
        HashFunctionPtr hashFunc;
        int N;
        int K;

    public:
        BaselineCountMinSketch(int n, int k, HashFunctionPtr func) : N(n), K(k), hashFunc(func) {
            estimates.resize(N, std::vector<uint32_t>(K, 0));
        }

        void insert (uint32_t item) {
            for (int i = 0; i < N; i++) {
                int bucket = hashFunc(item, i) % K;
                estimates[i][bucket] += 1;
            }
        }

        void remove (uint32_t item) {
            for (int i = 0; i < N; i++) {
                    int bucket = hashFunc(item, i) % K;
                    estimates[i][bucket] -= 1;
                }
        }

        uint32_t query_count (uint32_t item) {
            uint32_t estimate = UINT32_MAX;
            for (int i = 0; i < N; i++) {
                int bucket = hashFunc(item, i) % K;
                estimate = std::min(estimate, estimates[i][bucket]);
            }
            return estimate;
        }
};


//test
int main() {
    BaselineCountMinSketch cms(5, 100, MurmurHash3); 

    cms.insert(15);
    cms.insert(15);
    cms.insert(20);
    cms.remove(20);
    cms.insert(8);

    std::cout << "Count of 15: " << cms.query_count(15) << std::endl; // Should output 2
    std::cout << "Count of 20: " << cms.query_count(20) << std::endl; // Should output 0
    std::cout << "Count of 8:" << cms.query_count(8) << std::endl; //Should output 1
    std::cout << "Count of 1:" << cms.query_count(1) << std::endl; //Should output 0

    return 0;
}