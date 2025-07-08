// ***********************************************************************
// 
// Jet: Multilevel Graph Partitioning
//
// Copyright 2023 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS). 
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <utility>
#include <numeric>
#include <random>
#include <type_traits>
#include "metis.h"
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class initial_partitioner {
public:
    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using scalar_t = typename matrix_t::value_type;
    using edge_offset_t = typename matrix_t::size_type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using vtx_mt = typename vtx_vt::HostMirror;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using wgt_mt = typename wgt_vt::HostMirror;
    using edge_vt = Kokkos::View<edge_offset_t*, Device>;
    using edge_mt = typename edge_vt::HostMirror;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_mt = typename part_vt::HostMirror;
    using metis_int = int;
    using metis_vt = Kokkos::View<metis_int*, Device>;
    using metis_mt = typename metis_vt::HostMirror;
    using policy_t = Kokkos::RangePolicy<exec_space>;

template <class dst_vt, class src_vt>
static void copy(dst_vt dst, src_vt src){
    Kokkos::parallel_for("copy", policy_t(0, src.extent(0)), KOKKOS_LAMBDA(const int i){
        dst(i) = src(i);
    });
}

template <class src_vt>
static metis_mt to_metis_int(src_vt src){
    using src_t = std::remove_cv_t<typename src_vt::value_type>;
    int n = src.extent(0);
    metis_mt data(Kokkos::ViewAllocateWithoutInitializing("metis int host"), n);
    if(std::is_same_v<metis_int, src_t>){
        Kokkos::deep_copy(data, src);
    } else {
        metis_vt data_dev(Kokkos::ViewAllocateWithoutInitializing("metis int dev"), n);
        copy<metis_vt, src_vt>(data_dev, src);
        Kokkos::deep_copy(data, data_dev);
    }
    return data;
}

static part_vt metis_init(matrix_t g, wgt_vt vtx_w, int k, double imb_ratio){
    int n = g.numRows();
    metis_vt part_metis("part metis type", n);
    metis_mt pm = Kokkos::create_mirror_view(part_metis);
    metis_mt vtx_wm = to_metis_int<wgt_vt>(vtx_w);
    metis_mt xadj = to_metis_int<typename matrix_t::row_map_type>(g.graph.row_map);
    metis_mt adjcwgt = to_metis_int<wgt_vt>(g.values);
    metis_mt adjncy = to_metis_int<vtx_vt>(g.graph.entries);
    real_t imbalance = imb_ratio;
    int ec = 0;
    int nweights = 1;
    int ret = METIS_PartGraphKway(&n, &nweights, xadj.data(), adjncy.data(),
				       vtx_wm.data(), NULL, adjcwgt.data(), &k, NULL,
				       &imbalance, NULL, &ec, pm.data());
    if(ret != METIS_OK){
        std::cerr << "Metis could not partition coarsest graph. Exiting..." << std::endl;
        exit(-1);
    }
    Kokkos::deep_copy(part_metis, pm);
    part_vt part("part", n);
    copy<part_vt, metis_vt>(part, part_metis);
    return part;
}

static part_vt random_init(wgt_vt vtx_w, int k, double imb_ratio){
    ordinal_t n = vtx_w.extent(0);
    scalar_t total = 0;
    Kokkos::parallel_reduce("sum vtx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vtx_w(i);
    }, total);
    scalar_t opt = total / k;
    scalar_t upper = opt * imb_ratio;
    wgt_mt vw = Kokkos::create_mirror_view(vtx_w);
    Kokkos::deep_copy(vw, vtx_w);
    part_vt part_dev("part device", n);
    part_mt part = Kokkos::create_mirror_view(part_dev);
    wgt_mt psizes("part sizes", k);
    std::random_device rd;
    std::mt19937 rg(rd());
    std::uniform_int_distribution<> range(0, k-1);
    for(ordinal_t i = 0; i < n; i++){
        scalar_t size = vw(i);
        part_t p = range(rg);
        int breaker = 0;
        while(!(psizes(p) + size < upper) && breaker < 2*k){
            p = range(rg);
            breaker++;
        }
        //this loop is needed in case vtx i can't find a valid fit in 2*k attempts
        while(!(psizes(p) < opt)){
            p = range(rg);
        }
        part(i) = p;
        psizes(p) += size;
    }
    Kokkos::deep_copy(part_dev, part);
    return part_dev;
}

class maxheap {
    std::vector<ordinal_t> keys;
    std::vector<scalar_t> vals;
    std::vector<int> loc;
    int size;

    void swap(int l1, int l2){
        ordinal_t k1 = keys[l1];
        ordinal_t k2 = keys[l2];
        scalar_t v1 = vals[l1];
        scalar_t v2 = vals[l2];
        loc[k1] = l2;
        loc[k2] = l1;
        keys[l1] = k2;
        keys[l2] = k1;
        vals[l1] = v2;
        vals[l2] = v1;
    }

    void bubble_up(int l) {
        while(l - 1 >= 0){
            int parent = (l - 1) / 2;
            if(vals[l] > vals[parent]){
                swap(l, parent);
                l = parent;
            } else {
                break;
            }
        }
    }

    void bubble_down(int l) {
        while(l*2 + 1 < size){
            int left = l*2 + 1;
            int right = left + 1;
            int compare = left;
            if(right < size) {
                if(vals[right] > vals[left]){
                    compare = right;
                }
            }
            if(vals[compare] > vals[l]) {
                swap(l, compare);
                l = compare;
            } else {
                break;
            }
        }
    }

public:
    maxheap(ordinal_t n) :
        keys(n, -1),
        vals(n),
        loc(n, -1),
        size(0) {}

    void insert_or_update(ordinal_t v, scalar_t base, scalar_t add){
        int l = loc[v];
        if(l == -1){
            loc[v] = size++;
            l = loc[v];
            keys[l] = v;
            vals[l] = base;
        }
        vals[l] += add;
        bubble_up(l);
    }

    ordinal_t pop_top() {
        ordinal_t top = keys[0];
        loc[top] = -1;
        size--;
        if(size > 0){
            keys[0] = keys[size];
            vals[0] = vals[size];
            loc[keys[0]] = 0;
            bubble_down(0);
        }
        return top;
    }

    void clear(){
        for(int i = 0; i < size; i++){
            int k = keys[i];
            loc[k] = -1;
        }
        size = 0;
    }

    int get_size() {
        return size;
    }
};

class minheap {
    std::vector<ordinal_t> keys;
    std::vector<scalar_t> vals;
    std::vector<int> loc;
    int size;

    void swap(int l1, int l2){
        ordinal_t k1 = keys[l1];
        ordinal_t k2 = keys[l2];
        scalar_t v1 = vals[l1];
        scalar_t v2 = vals[l2];
        loc[k1] = l2;
        loc[k2] = l1;
        keys[l1] = k2;
        keys[l2] = k1;
        vals[l1] = v2;
        vals[l2] = v1;
    }

    void bubble_up(int l) {
        while(l - 1 >= 0){
            int parent = (l - 1) / 2;
            if(vals[l] < vals[parent]){
                swap(l, parent);
                l = parent;
            } else {
                break;
            }
        }
    }

    void bubble_down(int l) {
        while(l*2 + 1 < size){
            int left = l*2 + 1;
            int right = left + 1;
            int compare = left;
            if(right < size) {
                if(vals[right] < vals[left]){
                    compare = right;
                }
            }
            if(vals[compare] < vals[l]) {
                swap(l, compare);
                l = compare;
            } else {
                break;
            }
        }
    }

public:
    minheap(ordinal_t n) :
        keys(n, -1),
        vals(n, 0),
        loc(n, -1),
        size(0) {}

    void insert_or_update(ordinal_t v, scalar_t add){
        int l = loc[v];
        if(l == -1){
            loc[v] = size++;
            l = loc[v];
            keys[l] = v;
            vals[l] = 0;
        }
        vals[l] += add;
        bubble_down(l);
    }

    ordinal_t pop_top() {
        ordinal_t top = keys[0];
        loc[top] = -1;
        size--;
        if(size > 0){
            keys[0] = keys[size];
            vals[0] = vals[size];
            loc[keys[0]] = 0;
            bubble_down(0);
        }
        return top;
    }

    ordinal_t top() {
        return keys[0];
    }

    void clear(){
        for(int i = 0; i < size; i++){
            int k = keys[i];
            loc[k] = -1;
        }
        size = 0;
    }

    int get_size() {
        return size;
    }
};

static part_t argmin(wgt_mt part_size, int k){
    ordinal_t am = 0;
    scalar_t m = part_size(am);
    for(ordinal_t p = 1; p < k; p++){
        scalar_t s = part_size(p);
        if(s < m){
            m = s;
            am = p;
        }
    }
    return am;
}

static part_t max(wgt_mt part_size, int k){
    scalar_t m = 0;
    for(ordinal_t p = 0; p < k; p++){
        scalar_t s = part_size(p);
        if(s > m){
            m = s;
        }
    }
    return m;
}

static scalar_t cut(edge_mt row_map, vtx_mt entries, wgt_mt values, part_mt part, ordinal_t n){
    scalar_t c = 0;
    for(ordinal_t u = 0; u < n; u++){
        for(edge_offset_t j = row_map(u); j < row_map(u+1); j++){
            ordinal_t v = entries(j);
            if(part(u) != part(v)) c += values(j);
        }
    }
    return c;
}

static part_vt ggg(matrix_t g, wgt_vt vw_dev, int k, double imb_ratio) {
    ordinal_t n = g.numRows();
    scalar_t total = 0;
    Kokkos::parallel_reduce("sum vtx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vw_dev(i);
    }, total);
    scalar_t opt = total / k;
    scalar_t upper = opt * imb_ratio;
    edge_mt row_map("row map host", n+1);
    vtx_mt entries("entries host", g.nnz());
    wgt_mt values("values host", g.nnz());
    wgt_mt vw("vtx weights host", n);
    Kokkos::deep_copy(row_map, g.graph.row_map);
    Kokkos::deep_copy(entries, g.graph.entries);
    Kokkos::deep_copy(values, g.values);
    Kokkos::deep_copy(vw, vw_dev);
    part_mt part("partition host", n);
    part_mt best_part("best partition", n);
    scalar_t best_cut = 1000000000;
    wgt_mt part_size("part sizes", k);

    std::srand(std::time({}));
    std::vector<ordinal_t> order(n);
    for(ordinal_t i = 0; i < n; i++){
        order[i] = i;
    }

    std::vector<scalar_t> base_prio(n, 0);
    for(ordinal_t u = 0; u < n; u++){
        for(edge_offset_t j = row_map(u); j < row_map(u+1); j++){
            base_prio[u] -= values(j);
        }
    }

    maxheap h(n);
    minheap ps_h(k);

    for(int trial = 0; trial < 5; trial++){
        for(part_t p = 0; p < k; p++){
            ps_h.insert_or_update(p, 0);
        }
        Kokkos::deep_copy(part, -1);
        Kokkos::deep_copy(part_size, 0);
        for(int i = 0; i < n - 1; i++){
            int s = std::rand() % (n - 1 - i);
            s += i;
            ordinal_t c = order[s];
            order[s] = order[i];
            order[i] = c;
        }
        // could use permutation ordering here
        // maybe sort by vtx wgt?
        for(ordinal_t x = 0; x < n; x++){
            ordinal_t i = order[x];
            // i may already be assigned
            if(part(i) != -1) continue;
            // choose emptiest part
            part_t p = ps_h.top();
            h.insert_or_update(i, base_prio[i], 0);
            // while p has more room, grow partition from a pseudo-BFS around i
            // at least one iteration (ie. for vertex i) should always occur
            while(h.get_size() > 0){
                ordinal_t u = h.pop_top();
                if(part_size(p) + vw(u) >= upper && i != u) continue;
                // std::cout << "Adding vertex " << u << " to part " << p << std::endl;
                part(u) = p;
                part_size(p) += vw(u);
                ps_h.insert_or_update(p, vw(u));
                for(edge_offset_t j = row_map(u); j < row_map(u+1); j++){
                    ordinal_t v = entries(j);
                    // ignore assigned vertices
                    if(part(v) != -1) continue;
                    scalar_t wgt = values(j);
                    // std::cout << "Adding vertex " << v << " with wgt " << wgt << " to maxheap" << std::endl;
                    h.insert_or_update(v, base_prio[v], 2*wgt);
                }
            }
            h.clear();
        }
        scalar_t cutsize = cut(row_map, entries, values, part, n);
        std::cout << cutsize << std::endl;
        if(cutsize < best_cut){
            best_cut = cutsize;
            Kokkos::deep_copy(best_part, part);
        }
        ps_h.clear();
    }
    part_vt part_dev("part dev", n);
    Kokkos::deep_copy(part_dev, best_part);
    return part_dev;
}

};

}
