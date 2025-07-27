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
#pragma once
#include <list>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "memory_store.hpp"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include "coarse_level.h"

struct is_nonnegative {
    __host__ __device__
    bool operator()(const int x){
        return x >= 0;
    }
};

namespace jet_community {

template<typename ordinal_t>
KOKKOS_INLINE_FUNCTION ordinal_t xorshiftHash(ordinal_t key) {
  ordinal_t x = key;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

template<class crsMat> //typename ordinal_t, typename edge_offset_t, typename scalar_t, class Device>
class contracter {
public:

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using graph_type = typename matrix_t::staticcrsgraph_type;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using mem_t = jet_partitioner::memory_store<matrix_t, ordinal_t>;
    using coarse_level_t = coarse_level<matrix_t>;
    static constexpr ordinal_t get_null_val() {
        // this value must line up with the null value used by the hashmap
        // accumulator
        if (std::is_signed<ordinal_t>::value) {
            return -1;
        } else {
            return std::numeric_limits<ordinal_t>::max();
        }
    }
    static constexpr ordinal_t ORD_MAX  = get_null_val();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;

struct countingFunctor {

    matrix_t g;
    vtx_view_t vcmap;
    edge_view_t degree_initial;
    ordinal_t workLength;

    countingFunctor(matrix_t _g,
            vtx_view_t _vcmap,
            edge_view_t _degree_initial) :
        g(_g),
        vcmap(_vcmap),
        degree_initial(_degree_initial),
        workLength(_g.numRows()) {}

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const 
    {
        ordinal_t u = vcmap(i);
        edge_offset_t start = g.graph.row_map(i);
        edge_offset_t end = g.graph.row_map(i + 1);
        ordinal_t nonLoopEdgesTotal = end - start;
        Kokkos::atomic_add(&degree_initial(u), nonLoopEdgesTotal);
    }
};

template <bool uniform>
struct combineAndDedupe {
    matrix_t g;
    vtx_view_t vcmap;
    vtx_view_t htable;
    wgt_view_t hvals;
    edge_view_t hrow_map;
    edge_view_t counts;
    vtx_view_t vtx;

    combineAndDedupe(matrix_t _g,
            vtx_view_t _vcmap,
            vtx_view_t _htable,
            wgt_view_t _hvals,
            edge_view_t _hrow_map,
            edge_view_t _counts,
            vtx_view_t _vtx) :
            g(_g),
            vcmap(_vcmap),
            htable(_htable),
            hvals(_hvals),
            hrow_map(_hrow_map),
            counts(_counts),
            vtx(_vtx) {}

    KOKKOS_INLINE_FUNCTION
        edge_offset_t insert(const edge_offset_t& hash_start, const edge_offset_t& size, const ordinal_t& u, const ordinal_t& i) const {
            edge_offset_t offset = abs(xorshiftHash<ordinal_t>(u) % size);
            while(true){
                if(htable(hash_start + offset) == -1){
                    if(Kokkos::atomic_compare_exchange(&htable(hash_start + offset), -1, u) == -1){
                        Kokkos::atomic_add(&counts(i), 1);
                    }
                }
                if(htable(hash_start + offset) == u){
                    return offset;
                } else {
                    offset++;
                    if(offset >= size) offset -= size;
                }
            }
        }

    KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const
    {
        const ordinal_t x = vtx(thread.league_rank());
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i == u) return;
            edge_offset_t offset = insert(hash_start, size, u, i);
            if constexpr(uniform) Kokkos::atomic_add(&hvals(hash_start + offset), 1);
            else Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
        });
    }

    KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& xx) const
    {
        const ordinal_t x = vtx(xx);
        const ordinal_t i = vcmap(x);
        const edge_offset_t start = g.graph.row_map(x);
        const edge_offset_t end = g.graph.row_map(x + 1);
        const edge_offset_t hash_start = hrow_map(i);
        const edge_offset_t size = hrow_map(i + 1) - hash_start;
        for(edge_offset_t j = start; j < end; j++){
            ordinal_t u = vcmap(g.graph.entries(j));
            if(i == u) continue;
            edge_offset_t offset = insert(hash_start, size, u, i);
            if constexpr(uniform) Kokkos::atomic_add(&hvals(hash_start + offset), 1);
            else Kokkos::atomic_add(&hvals(hash_start + offset), g.values(j));
        }
    }
};

void fast_fill(vtx_view_t a, ordinal_t V){
    edge_offset_t width = a.extent(0);
    edge_offset_t w8 = (width + 7)/8;
    Kokkos::parallel_for("fast fill", policy_t(0, w8), KOKKOS_LAMBDA(const edge_offset_t i){
        a(i) = V;
        a(i + w8) = V;
        a(i + 2*w8) = V;
        a(i + 3*w8) = V;
        a(i + 4*w8) = V;
        a(i + 5*w8) = V;
        a(i + 6*w8) = V;
        // this one may be out of bounds for even arbitrarily large width
        // technically some of the other ones can also be out of bounds for width < 42
        if(i + 7*w8 < width) a(i + 7*w8) = V;
    });
}

template <bool uniform>
coarse_level_t build_coarse_graph(const coarse_level_t level,
    const vtx_view_t vcmap,
    const ordinal_t nc,
    mem_t& mem) {

    matrix_t g = level.mtx;
    ordinal_t n = g.numRows();

    Kokkos::Timer timer;
    edge_view_t hrow_map = Kokkos::subview(mem.p_mem.row_map, std::make_pair((ordinal_t)0, nc + 1));
    Kokkos::deep_copy(exec_space(), hrow_map, 0);
    countingFunctor countF(g, vcmap, hrow_map);
    Kokkos::parallel_for("count edges per coarse vertex (also compute coarse vertex weights)", policy_t(0, n), countF);
    edge_offset_t hash_size = 0;
    //exclusive prefix sum
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = hrow_map(i);
        if(final){
            hrow_map(i) = update;
        }
        update += val;
    }, mem.s_mem.scan_host);
    exec_space().fence();
    hash_size = mem.s_mem.scan_host();
    vtx_view_t htable = Kokkos::subview(mem.p_mem.entries, std::make_pair((edge_offset_t)0, hash_size));
    fast_fill(htable, -1);
    // Kokkos::deep_copy(exec_space(), htable, -1);
    wgt_view_t hvals = Kokkos::subview(mem.p_mem.vals, std::make_pair((edge_offset_t)0, hash_size));
    Kokkos::deep_copy(exec_space(), hvals, 0);
    //insert each coarse vertex into a bucket determined by a hash
    //use linear probing to resolve conflicts
    //combine weights using atomic addition
    edge_view_t coarse_row_map_f("edges_per_source", nc + 1);
    ordinal_t low = mem.p_mem.offset_mid;
    ordinal_t high = n - low;
    vtx_view_t vtx_high = Kokkos::subview(mem.p_mem.order2, std::make_pair(low, n));
    vtx_view_t vtx_low = Kokkos::subview(mem.p_mem.order2, std::make_pair(static_cast<ordinal_t>(0), low));
    combineAndDedupe<uniform> cnd_low(g, vcmap, htable, hvals, hrow_map, coarse_row_map_f, vtx_low);
    combineAndDedupe<uniform> cnd_high(g, vcmap, htable, hvals, hrow_map, coarse_row_map_f, vtx_high);
    Kokkos::parallel_for("deduplicate", team_policy_t(high, Kokkos::AUTO), cnd_high);
    Kokkos::parallel_for("deduplicate", policy_t(0, low), cnd_low);
    edge_offset_t old_size = hash_size;
    Kokkos::parallel_scan("scan offsets", policy_t(0, nc + 1), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update, const bool final){
        edge_offset_t val = coarse_row_map_f(i);
        if(final){
            coarse_row_map_f(i) = update;
        }
        update += val;
    }, mem.s_mem.scan_host);
    exec_space().fence();
    hash_size = mem.s_mem.scan_host();
    vtx_view_t entries_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse entries"), hash_size);
    wgt_view_t wgts_coarse(Kokkos::ViewAllocateWithoutInitializing("coarse weights"), hash_size);
    Kokkos::fence();
    thrust::device_ptr<int> htb(htable.data());
    thrust::counting_iterator<int> iter(0);
    thrust::device_ptr<int> ec(entries_coarse.data());
    thrust::copy_if(thrust::device, iter, iter + old_size, htb, ec, is_nonnegative());
    Kokkos::parallel_for("read", policy_t(0, hash_size), KOKKOS_LAMBDA(const edge_offset_t j){
        edge_offset_t jx = entries_coarse(j);
        entries_coarse(j) = htable(jx);
        wgts_coarse(j) = hvals(jx);
    });
    graph_type gc_graph(entries_coarse, coarse_row_map_f);
    matrix_t gc("gc", nc, wgts_coarse, gc_graph);
    coarse_level_t next_level;
    next_level.mtx = gc;
    return next_level;
}

};

}
