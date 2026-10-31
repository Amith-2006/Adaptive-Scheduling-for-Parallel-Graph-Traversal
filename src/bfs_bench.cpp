
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include "partition.h"


using Graph = std::vector<std::vector<int>>;


struct Stats {
    double mean = 0;
    double variance = 0;
    double skew = 0;
    double cv = 0;
    long long edges = 0;
};

Graph read_graph(const std::string& path) {
    std::ifstream in(path);

    if (!in) {
        throw std::runtime_error("cannot open graph: " + path);
    }

    int n, m;

    if (!(in >> n >> m) || n <= 0 || m < 0) {
        throw std::runtime_error("invalid graph header");
    }

    Graph g(static_cast<size_t>(n));

    for (int i = 0; i < m; ++i) {
        int u, v;

        if (!(in >> u >> v)) {
            throw std::runtime_error("invalid edge data");
        }

        if (u < 0 || u >= n || v < 0 || v >= n) {
            throw std::runtime_error("edge endpoint out of range");
        }

        if (u == v) {
            continue;
        }

        g[u].push_back(v);
        g[v].push_back(u);
    }

    for (auto& neighbors : g) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(
            std::unique(neighbors.begin(), neighbors.end()),
            neighbors.end()
        );
    }

    return g;
}

std::vector<int> sequential_bfs(const Graph& g) {
    const int n = static_cast<int>(g.size());

    std::vector<int> dist(static_cast<size_t>(n), -1);
    std::queue<int> q;

    dist[0] = 0;
    q.push(0);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : g[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    return dist;
}

Stats frontier_stats(const Graph& g, const std::vector<int>& frontier) {
    Stats s;

    if (frontier.empty()) {
        return s;
    }

    std::vector<double> degrees;
    degrees.reserve(frontier.size());

    for (int u : frontier) {
        double degree = static_cast<double>(g[u].size());
        degrees.push_back(degree);
        s.edges += static_cast<long long>(g[u].size());
    }

    s.mean = std::accumulate(
        degrees.begin(), degrees.end(), 0.0
    ) / static_cast<double>(degrees.size());

    double m2 = 0;
    double m3 = 0;

    for (double x : degrees) {
        double z = x - s.mean;
        m2 += z * z;
        m3 += z * z * z;
    }

    double count = static_cast<double>(degrees.size());

    s.variance = m2 / count;

    double sd = std::sqrt(s.variance);

    s.cv = s.mean != 0 ? sd / s.mean : 0;

    s.skew = sd > 0
        ? (m3 / count) / (sd * sd * sd)
        : 0;

    return s;
}

struct Run {
    std::vector<int> dist;
    double seconds = 0;
    std::vector<std::string> rows;
};

Run bfs(
    const Graph& g,
    const std::string& scheduler,
    int threads,
    int trial
) {
    const int n = static_cast<int>(g.size());

    Run out;
    out.dist.assign(static_cast<size_t>(n), -1);
    out.dist[0] = 0;

    std::vector<int> frontier{0};
    int level = 0;

    while (!frontier.empty()) {
        Stats st = frontier_stats(g, frontier);

        const double t0 = omp_get_wtime();

        int nt = std::max(1, threads);

        std::vector<std::vector<int>> local(
            static_cast<size_t>(nt)
        );

        std::vector<int> newly_seen(static_cast<size_t>(n), 0);
        std::vector<int> next;

        if (scheduler == "edge_balanced") {
            std::vector<long long> prefix(frontier.size() + 1, 0);

            for (size_t i = 0; i < frontier.size(); ++i) {
                prefix[i + 1] = prefix[i] +
                    static_cast<long long>(g[frontier[i]].size());
            }


            #pragma omp parallel num_threads(nt)
{
    int tid = omp_get_thread_num();
    int actual_threads = omp_get_num_threads();

    std::vector<size_t> boundaries =
        calculate_boundaries(
            prefix,
            actual_threads,
            frontier.size()
        );

    size_t a = boundaries[tid];
    size_t b = boundaries[tid + 1];

    for (size_t i = a; i < b; ++i) {
        for (int v : g[frontier[i]]) {
            if (out.dist[v] == -1) {
                local[tid].push_back(v);
            }
        }
    }
}
        } else if (scheduler == "static") {
            #pragma omp parallel for schedule(static) num_threads(nt)
            for (long long i = 0;
                 i < static_cast<long long>(frontier.size());
                 ++i) {
                int tid = omp_get_thread_num();

                for (int v : g[frontier[static_cast<size_t>(i)]]) {
                    if (out.dist[v] == -1) {
                        local[tid].push_back(v);
                    }
                }
            }
        } else if (scheduler == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
            for (long long i = 0;
                 i < static_cast<long long>(frontier.size());
                 ++i) {
                int tid = omp_get_thread_num();

                for (int v : g[frontier[static_cast<size_t>(i)]]) {
                    if (out.dist[v] == -1) {
                        local[tid].push_back(v);
                    }
                }
            }
        } else if (scheduler == "guided") {
            #pragma omp parallel for schedule(guided, 1) num_threads(nt)
            for (long long i = 0;
                 i < static_cast<long long>(frontier.size());
                 ++i) {
                int tid = omp_get_thread_num();

                for (int v : g[frontier[static_cast<size_t>(i)]]) {
                    if (out.dist[v] == -1) {
                        local[tid].push_back(v);
                    }
                }
            }
        } else {
            throw std::runtime_error(
                "unknown scheduler: " + scheduler
            );
        }

        for (const auto& bucket : local) {
            for (int v : bucket) {
                if (out.dist[v] == -1 && !newly_seen[v]) {
                    newly_seen[v] = 1;
                    next.push_back(v);
                }
            }
        }

        for (int v : next) {
            out.dist[v] = level + 1;
        }

        double elapsed = omp_get_wtime() - t0;

        std::ostringstream row;

        row << trial << ','
            << scheduler << ','
            << level << ','
            << frontier.size() << ','
            << st.edges << ','
            << std::setprecision(10)
            << st.mean << ','
            << st.variance << ','
            << st.skew << ','
            << st.cv << ','
            << elapsed << '\n';

        out.rows.push_back(row.str());
        out.seconds += elapsed;

        frontier.swap(next);
        ++level;
    }

    return out;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr
            << "Usage: bfs_bench GRAPH.edgelist THREADS OUT.csv REPEATS\n";
        return 2;
    }

    try {
        std::string path = argv[1];
        std::string outpath = argv[3];

        int threads = std::stoi(argv[2]);
        int repeats = std::stoi(argv[4]);

        if (threads <= 0 || repeats <= 0) {
            throw std::runtime_error(
                "THREADS and REPEATS must be positive"
            );
        }

        Graph g = read_graph(path);

        const std::vector<int> reference = sequential_bfs(g);

        std::vector<std::string> schedulers = {
            "static",
            "dynamic",
            "guided",
            "edge_balanced"
        };

        std::ofstream out(outpath);

        if (!out) {
            throw std::runtime_error(
                "cannot write output: " + outpath
            );
        }

        out
            << "trial,scheduler,level,frontier_size,frontier_edges,"
            << "degree_mean,degree_variance,degree_skew,degree_cv,"
            << "level_seconds\n";

        for (int r = 0; r < repeats; ++r) {
            for (size_t i = 0; i < schedulers.size(); ++i) {
                const size_t index = (i + static_cast<size_t>(r))
                    % schedulers.size();

                const std::string& scheduler = schedulers[index];

                Run result = bfs(g, scheduler, threads, r);

                if (result.dist != reference) {
                    throw std::runtime_error(
                        "BFS correctness mismatch for scheduler " +
                        scheduler
                    );
                }

                for (const auto& line : result.rows) {
                    out << line;
                }

                std::cerr
                    << "trial=" << r
                    << " scheduler=" << scheduler
                    << " total_level_seconds="
                    << std::setprecision(8)
                    << result.seconds << '\n';
            }
        }

        out.flush();

        if (!out) {
            throw std::runtime_error("failed while writing CSV output");
        }

        std::cerr
            << "PASS: all scheduler distances match; vertices="
            << g.size() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
