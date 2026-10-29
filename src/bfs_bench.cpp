#include <algorithm>
#include <cmath>
#include <cstdint>
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

#ifdef USE_TBB
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/partitioner.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#endif

using Graph = std::vector<std::vector<int>>;
struct Stats { double mean=0, variance=0, skew=0, cv=0; long long edges=0; };

Graph read_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open graph: " + path);
    int n; long long m;
    if (!(in >> n >> m) || n < 1 || m < 0) throw std::runtime_error("invalid graph header");
    Graph g(static_cast<size_t>(n));
    int u,v;
    while (in >> u >> v) {
        if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;
        g[u].push_back(v); g[v].push_back(u);
    }
    for (auto& a : g) { std::sort(a.begin(), a.end()); a.erase(std::unique(a.begin(), a.end()), a.end()); }
    return g;
}
Stats frontier_stats(const Graph& g, const std::vector<int>& f) {
    Stats s; if (f.empty()) return s;
    std::vector<double> d; d.reserve(f.size());
    for (int u : f) { d.push_back(static_cast<double>(g[u].size())); s.edges += static_cast<long long>(g[u].size()); }
    s.mean = std::accumulate(d.begin(), d.end(), 0.0) / d.size();
    double m2=0, m3=0;
    for (double x : d) { double z=x-s.mean; m2 += z*z; m3 += z*z*z; }
    s.variance=m2/d.size();
    double sd=std::sqrt(s.variance);
    s.cv = s.mean != 0 ? sd/s.mean : 0;
    s.skew = sd > 0 ? (m3/d.size())/(sd*sd*sd) : 0;
    return s;
}

struct Run { std::vector<int> dist; double seconds=0; std::vector<std::string> rows; };
Run bfs(const Graph& g, const std::string& scheduler, int threads, int trial) {
    const int n=static_cast<int>(g.size());
    Run out; out.dist.assign(n,-1); out.dist[0]=0;
    std::vector<int> frontier{0}; int level=0;
    while (!frontier.empty()) {
        Stats st=frontier_stats(g,frontier);
        const double t0=omp_get_wtime();
        int nt=std::max(1,threads);
        std::vector<std::vector<int>> local(static_cast<size_t>(nt));
        std::vector<int> newly_seen(static_cast<size_t>(n),0);
        std::vector<int> next;
#ifdef USE_TBB
        if (scheduler == "tbb") {
            oneapi::tbb::enumerable_thread_specific<std::vector<int>> tbb_local;
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, frontier.size(), 64), [&](const oneapi::tbb::blocked_range<size_t>& range) {
                auto& bucket=tbb_local.local();
                for(size_t i=range.begin(); i<range.end(); ++i)
                    for(int v:g[frontier[i]]) if(out.dist[v]==-1) bucket.push_back(v);
            }, oneapi::tbb::auto_partitioner{});
            for(auto& bucket:tbb_local) local.push_back(std::move(bucket));
        } else
#endif
        if (scheduler == "edge_balanced") {
            std::vector<long long> prefix(frontier.size()+1,0);
            for (size_t i=0;i<frontier.size();++i) prefix[i+1]=prefix[i]+static_cast<long long>(g[frontier[i]].size());
            const long long total=prefix.back();
            #pragma omp parallel num_threads(nt)
            {
                int tid=omp_get_thread_num();
                long long lo=total*tid/nt, hi=total*(tid+1)/nt;
                auto begin=std::lower_bound(prefix.begin()+1,prefix.end(),lo+1)-prefix.begin()-1;
                auto end=std::lower_bound(prefix.begin()+1,prefix.end(),hi+1)-prefix.begin()-1;
                size_t a=static_cast<size_t>(std::min<long long>(begin,frontier.size()));
                size_t b=static_cast<size_t>(std::min<long long>(std::max<long long>(end+1,begin),frontier.size()));
                for(size_t i=a;i<b;i++) for(int v:g[frontier[i]]) if(out.dist[v]==-1) local[tid].push_back(v);
            }
        } else if (scheduler == "static") {
            #pragma omp parallel for schedule(static) num_threads(nt)
            for (long long i=0;i<static_cast<long long>(frontier.size());++i) { int tid=omp_get_thread_num(); for(int v:g[frontier[static_cast<size_t>(i)]]) if(out.dist[v]==-1) local[tid].push_back(v); }
        } else if (scheduler == "dynamic") {
            #pragma omp parallel for schedule(dynamic,1) num_threads(nt)
            for (long long i=0;i<static_cast<long long>(frontier.size());++i) { int tid=omp_get_thread_num(); for(int v:g[frontier[static_cast<size_t>(i)]]) if(out.dist[v]==-1) local[tid].push_back(v); }
        } else if (scheduler == "guided") {
            #pragma omp parallel for schedule(guided,1) num_threads(nt)
            for (long long i=0;i<static_cast<long long>(frontier.size());++i) { int tid=omp_get_thread_num(); for(int v:g[frontier[static_cast<size_t>(i)]]) if(out.dist[v]==-1) local[tid].push_back(v); }
        }
        // A deterministic serial deduplication phase keeps this baseline race-free.
        // It is included in the measured level time for all schedulers.
        for (const auto& bucket:local) for(int v:bucket) if(out.dist[v]==-1 && !newly_seen[v]) { newly_seen[v]=1; next.push_back(v); }
        for(int v:next) out.dist[v]=level+1;
        double elapsed=omp_get_wtime()-t0;
        std::ostringstream row;
        row << trial << ',' << scheduler << ',' << level << ',' << frontier.size() << ',' << st.edges << ','
            << std::setprecision(10) << st.mean << ',' << st.variance << ',' << st.skew << ',' << st.cv << ',' << elapsed << '\n';
        out.rows.push_back(row.str()); out.seconds += elapsed;
        frontier.swap(next); ++level;
    }
    return out;
}

int main(int argc,char** argv) {
    if(argc<5) { std::cerr << "Usage: bfs_bench GRAPH.edgelist THREADS OUT.csv REPEATS\n"; return 2; }
    try {
        std::string path=argv[1], outpath=argv[3]; int threads=std::stoi(argv[2]), repeats=std::stoi(argv[4]);
        Graph g=read_graph(path);
        std::vector<std::string> schedulers={"static","dynamic","guided","edge_balanced"};
#ifdef USE_TBB
        schedulers.push_back("tbb");
#endif
        std::ofstream out(outpath); if(!out) throw std::runtime_error("cannot write output: "+outpath);
        out << "trial,scheduler,level,frontier_size,frontier_edges,degree_mean,degree_variance,degree_skew,degree_cv,level_seconds\n";
        std::vector<int> reference;
        for(int r=0;r<repeats;r++) for(const auto& s:schedulers) {
            Run result=bfs(g,s,threads,r);
            if(reference.empty()) reference=result.dist;
            else if(result.dist!=reference) throw std::runtime_error("BFS correctness mismatch for scheduler "+s);
            for(const auto& line:result.rows) out << line;
            std::cerr << "trial="<<r<<" scheduler="<<s<<" total_level_seconds="<<std::setprecision(8)<<result.seconds<<"\n";
        }
        std::cerr << "PASS: all scheduler distances match; vertices="<<g.size()<<"\n";
    } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; return 1; }
    return 0;
}
