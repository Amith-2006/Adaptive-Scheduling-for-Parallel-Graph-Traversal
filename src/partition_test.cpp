#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "partition.h"

using namespace std;


bool test_partition(
    const vector<long long>& degrees,
    int threads
) {
    if (threads <= 0) {
        return false;
    }

    for (long long degree : degrees) {
        if (degree < 0) {
            return false;
        }
    }

    vector<long long> prefix(degrees.size() + 1, 0);

for (size_t i = 0; i < degrees.size(); ++i) {
    prefix[i + 1] = prefix[i] + degrees[i];
}

vector<size_t> boundaries =
    calculate_boundaries(prefix, threads, degrees.size());

    size_t n = degrees.size();

    if (boundaries.front() != 0 ||
        boundaries.back() != n) {
        return false;
    }

    vector<int> ownership(n, 0);
    long long scanned_edges = 0;

    for (int t = 0; t < threads; ++t) {
        if (boundaries[t] > boundaries[t + 1]) {
            return false;
        }

        for (size_t i = boundaries[t];
             i < boundaries[t + 1]; ++i) {
            ownership[i]++;
            scanned_edges += degrees[i];
        }
    }

    for (int count : ownership) {
        if (count != 1) {
            return false;
        }
    }

    long long expected_edges =
        accumulate(degrees.begin(), degrees.end(), 0LL);

    return scanned_edges == expected_edges;
}

int main() {
    vector<vector<long long>> tests = {
        {3, 0, 5, 2},
        {1000, 1, 1, 1},
        {0, 0, 0, 0},
        {5},
        {1, 1},
        {10, 0, 0, 10},
        {1, 100, 1, 100, 1},
        {2, 2, 2, 2, 2, 2}
    };

    for (size_t i = 0; i < tests.size(); ++i) {
        if (!test_partition(tests[i], 4)) {
            cout << "FAIL: fixed test " << i + 1 << '\n';
            return 1;
        }

        cout << "PASS: fixed test " << i + 1 << '\n';
    }

    mt19937 rng(42);
    uniform_int_distribution<int> size_dist(0, 500);
    uniform_int_distribution<int> degree_dist(0, 1000);
    uniform_int_distribution<int> thread_dist(1, 16);

    for (int test = 1; test <= 1000; ++test) {
        int n = size_dist(rng);
        int threads = thread_dist(rng);

        vector<long long> degrees(n);

        for (int i = 0; i < n; ++i) {
            degrees[i] = degree_dist(rng);
        }

        if (!test_partition(degrees, threads)) {
            cout << "FAIL: random test " << test << '\n';
            return 1;
        }
    }

    cout << "PASS: 1000 randomized tests\n";
    cout << "All partition tests passed.\n";

    return 0;
}