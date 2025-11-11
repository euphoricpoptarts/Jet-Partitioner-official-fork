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
#include <type_traits>
#include <limits>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "memory_store.hpp"
#include "cluster_data.h"

namespace jet_community {

template<class crsMat>
class jet_refiner_cluster {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using obj_vt = Kokkos::View<float*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using refine_data = cluster_data<matrix_t>;
    using mem_t = jet_partitioner::memory_store<matrix_t, ordinal_t>;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr float OBJ_MIN = std::numeric_limits<float>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr ordinal_t NULL_PART = -1;
    static constexpr ordinal_t HASH_RECLAIM = -2;
    static constexpr ordinal_t NO_MOVE = -3;
    static constexpr ordinal_t MID_CUTOFF = 32;
    static constexpr ordinal_t LARGE_CUTOFF = 128;

    static const ordinal_t max_sections = 32;
    static const int max_buckets = 100;
    static const int mid_bucket = 50;

    static KOKKOS_INLINE_FUNCTION uint32_t hash(uint32_t x) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }

struct problem {
    matrix_t g;
    wgt_view_t vtx_w;
    wgt_view_t wdeg;
    bool use_team = true;
};

// vertex-part connectivity datastructure
struct cdata_t {
    edge_view_t conn_offsets;
    gain_vt conn_vals;
    vtx_vt conn_entries;
    // matrix wrapper of above views
    matrix_t c_graph;
    vtx_vt conn_table_sizes;
    bool init = false;
};

void relabel_contiguously(vtx_vt labels, refine_data& rfd, mem_t& mem){
	ordinal_t n = labels.extent(0);
    ordinal_t initial_count = rfd.label_count;
	vtx_vt used = Kokkos::subview(mem.s_mem.vtx1, std::make_pair((ordinal_t)0, initial_count));
    Kokkos::deep_copy(exec_space(), used, 0);
    // some vertices can have zero degree so some labels can have zero total degree
    // therefore we can't use rfd.total_deg to determine which labels are in use
	Kokkos::parallel_for("mark labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		used(labels(i)) = 1;
	});
    ordinal_t t_labels = 0;
	Kokkos::parallel_scan("count labels", policy_t(0, initial_count), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
		if(used(i) > 0){
			if(final) used(i) = update;
			update++;
		}
	}, t_labels);
	Kokkos::parallel_for("relabel", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		labels(i) = used(labels(i));
	});
    wgt_view_t total_deg("new total degree", t_labels);
    wgt_view_t cluster_size("new cluster size", t_labels);
    wgt_view_t old_total_deg = rfd.total_deg;
    wgt_view_t old_cluster_size = rfd.cluster_size;
    Kokkos::parallel_for("relabel degrees", policy_t(0, initial_count), KOKKOS_LAMBDA(const ordinal_t i){
		if(old_cluster_size(i) > 0){
            ordinal_t relabeled = used(i);
            total_deg(relabeled) = old_total_deg(i);
            cluster_size(relabeled) = old_cluster_size(i);
        }
	});
    rfd.total_deg = total_deg;
    rfd.cluster_size = cluster_size;
    rfd.label_count = t_labels;
}

KOKKOS_INLINE_FUNCTION
static ordinal_t gain_bucket(const gain_t& gx, const scalar_t& vwgt){
    //cast to float so we can approximate log_1.5
    float gain = static_cast<float>(gx) / static_cast<float>(vwgt);
    ordinal_t gain_type = 0;
    if(gain > 0.0){
        gain_type = 0;
    } else if(gain == 0.0) {
        gain_type = 1;
    } else {
        gain_type = mid_bucket;
        gain = abs(gain);
        if(gain < 1.0){
            while(gain < 1.0){
                gain *= 1.5;
                gain_type--;
            }
            if(gain_type < 2){
                gain_type = 2;
            }
        } else {
            while(gain > 1.0){
                gain /= 1.5;
                gain_type++;
            }
            if(gain_type >= max_buckets){
                gain_type = max_buckets - 1;
            }
        }
    }
    return gain_type;
}

//determines which vertices (if any) should be moved to another part to improve objective
//8 kernels, 2 device-host syncs
template <bool uniform>
vtx_vt jet_lp(const problem& prob, const matrix_t& c_graph, const vtx_vt& part, const refine_data& rfd, mem_t& mem, float filter_ratio, const vtx_vt vtx_constraint, const vtx_vt c_constraint){
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    vtx_vt dest_part = mem.p_mem.dest_part;
    obj_vt save_gains = mem.p_mem.obj_persistent;
    vtx_vt swap_bit = mem.s_mem.zeros1;
    gain_vt total_deg = rfd.total_deg;
    gain_vt wdeg = prob.wdeg;
    gain_vt pvals = mem.p_mem.pvals;
    float penalty_mod = rfd.get_penalty_modifier();
    vtx_vt vtx1 = mem.s_mem.vtx1;
    vtx_vt vtx2 = mem.s_mem.vtx2;
    vtx_vt order1 = mem.p_mem.order1;
    vtx_vt dead_bit = mem.p_mem.lock_bit;
    ordinal_t big_begin = mem.p_mem.offset_large;
    vtx_vt small_tables = Kokkos::subview(order1, std::make_pair(static_cast<ordinal_t>(0), big_begin));
    vtx_vt large_tables = Kokkos::subview(order1, std::make_pair(big_begin, n));
    // if input label count is smaller than LARGE_CUTOFF, then all tables are also smaller than LARGE_CUTOFF
    bool truncated = (rfd.label_count <= LARGE_CUTOFF);
    Kokkos::parallel_for("select destination part (small tables)", policy_t(0, truncated ? n : big_begin), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = truncated ? x : small_tables(x);
        if(dest_part(i) != NULL_PART){
            return;
        }
        ordinal_t best = NO_MOVE;
        float wd = wdeg(i);
        float multi = wd*penalty_mod;
        ordinal_t p = part(i);
        float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
        // b_conn must be at least this value to pass filter
        float b_conn = p_conn - filter_ratio*(p_conn);
        edge_offset_t start = c_graph.graph.row_map(i);
        edge_offset_t end = c_graph.graph.row_map(i+1);
        //finds potential destination as most connected part excluding p
        for(edge_offset_t j = start; j < end; j++){
            gain_t j_val = c_graph.values(j);
            if(j_val > 0 && j_val >= b_conn){
                ordinal_t px = c_graph.graph.entries(j);
                if(dead_bit(px)) continue;
                if(vtx_constraint(i) != c_constraint(px)) continue;
                float j_conn = j_val - static_cast<float>(total_deg(px))*multi;
                if(j_conn >= b_conn){
                    b_conn = j_conn;
                    best = px;
                }
            }
        }
        float gain = OBJ_MIN;
        if(best != NO_MOVE){
            // vertices must pass this filter in order to be considered further
            gain = b_conn - p_conn;
        }
        save_gains(i) = gain;
        //a vertex is not considered further if best == p
        dest_part(i) = best;
    });
    if(!truncated){
        Kokkos::parallel_for("select destination part (large tables)", team_policy_t(n - big_begin, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
            ordinal_t i = large_tables(t.league_rank());
            if(dest_part(i) != NULL_PART){
                return;
            }
            ordinal_t team_size = t.team_size();
            float wd = wdeg(i);
            float multi = wd*penalty_mod;
            edge_offset_t start = c_graph.graph.row_map(i);
            edge_offset_t end = c_graph.graph.row_map(i+1);
            ordinal_t p = part(i);
            float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
            // j_conn must be at least this value to pass filter
            float maxl = p_conn - filter_ratio*(p_conn);
            ordinal_t argmax = NO_MOVE;
            //finds potential destination as most connected part excluding p
            for(edge_offset_t j = start + t.team_rank(); j < end; j += team_size){
                gain_t j_val = c_graph.values(j);
                if(j_val > 0 && j_val >= maxl){
                    ordinal_t px = c_graph.graph.entries(j);
                    if(dead_bit(px)) continue;
                    if(vtx_constraint(i) != c_constraint(px)) continue;
                    float j_conn = j_val - static_cast<float>(total_deg(px))*multi;
                    if(j_conn >= maxl){
                        // this is not deterministic unless the case j_conn == maxl is handled properly
                        argmax = px;
                        maxl = j_conn;
                    }
                }
            }
            if(argmax == NO_MOVE) maxl = OBJ_MIN;
            float maxg = 0;
            float oldmaxl = maxl;
            t.team_reduce(Kokkos::Max<float, mem_space>(maxg), maxl);
            if(maxg == OBJ_MIN){
                if(t.team_rank() == 0){
                    dest_part(i) = NO_MOVE;
                    save_gains(i) = OBJ_MIN;
                }
                return;
            }
            if(oldmaxl != maxg) argmax = n + 1;
            ordinal_t argmaxg = NO_MOVE;
            t.team_reduce(Kokkos::Min<ordinal_t, mem_space>(argmaxg), argmax);
            if(t.team_rank() == 0){
                save_gains(i) = maxg - p_conn;
                dest_part(i) = argmaxg;
            }
        });
    }
    //need to store the pre-afterburn gains into a separate view
    //than savegains, because we write new values into it that may not be overwritten
    //if a vertex has its best neighbor cached
    obj_vt pregain = save_gains;
    vtx_pin_st pin_host = mem.s_mem.pin_host;
    // write all unlocked vertices that passed the above filter into an unordered list
    // output count of such vertices into num_pos
    // order1 is already organized into two buckets by degree > or <= 128
    Kokkos::parallel_scan("filter potentially viable moves", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update, const bool final){
        if(final && x == big_begin){
            pin_host() = update;
        }
        ordinal_t i = order1(x);
        ordinal_t best = dest_part(i);
        if(best != NO_MOVE && dead_bit(i) == 0){
            if(final){
                vtx1(update) = i;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    num_pos = mem.s_mem.scan_host();
    //truncate scratch views by num_pos
    vtx_vt pos_moves = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    if(big_begin < n){
        big_begin = pin_host();
    } else {
        big_begin = num_pos;
    }
    ordinal_t small = big_begin;
    ordinal_t big = num_pos - small;
    vtx_vt big_rows = Kokkos::subview(vtx1, std::make_pair(big_begin, num_pos));
    vtx_vt small_rows = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), small));
    float eps = 0.1;
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(big, 256), KOKKOS_LAMBDA(const member& t){
        float change = 0;
        ordinal_t i = big_rows(t.league_rank());
        ordinal_t best = dest_part(i);
        ordinal_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*penalty_mod;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, float& update){
            ordinal_t v = g.graph.entries(j);
            if(dead_bit(v)) return;
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                ordinal_t vpart = dest_part(v);
                scalar_t wgt;
                if constexpr(uniform) wgt = 1;
                else wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                update -= (vpart == p) ? q : 0;
                update += (vpart == best) ? q : 0;
                vpart = part(v);
                update += (vpart == p) ? q : 0;
                update -= (vpart == best) ? q : 0;
            }
        }, change);
        if(t.team_rank() == 0){
            if(igain + change >= 0){
                swap_bit(i) = 1;
            }
        }
    });
    Kokkos::parallel_for("afterburner heuristic", policy_t(0, small), KOKKOS_LAMBDA(const ordinal_t& x){
        float change = 0;
        ordinal_t i = small_rows(x);
        ordinal_t best = dest_part(i);
        ordinal_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*penalty_mod;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            if(dead_bit(v)) continue;
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                ordinal_t vpart = dest_part(v);
                scalar_t wgt;
                if constexpr(uniform) wgt = 1;
                else wgt = g.values(j);
                float q = static_cast<float>(wgt) - multi*prob.wdeg(v);
                change -= (vpart == p) ? q : 0;
                change += (vpart == best) ? q : 0;
                vpart = part(v);
                change += (vpart == p) ? q : 0;
                change -= (vpart == best) ? q : 0;
            }
        }
        if(igain + change >= 0){
            swap_bit(i) = 1;
        }
    });
    vtx_vt swaps2 = Kokkos::subview(vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    //scan all vertices that passed the post filter
    Kokkos::parallel_scan("filter beneficial moves", policy_t(0, num_pos), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(swap_bit(pos_moves(i))){
            if(final){
                swaps2(update) = pos_moves(i);
                // reset to zero for later use
                swap_bit(pos_moves(i)) = 0;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    num_pos = mem.s_mem.scan_host();
    pos_moves = Kokkos::subview(swaps2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    return pos_moves;
}

void count_oversized(ordinal_t n, wgt_view_t cluster_size, gain_t upper_bound){
    ordinal_t total_oversized = 0;
    Kokkos::parallel_reduce("compute oversized idx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update){
        if(cluster_size(i) > upper_bound){
            update++;
        }
    }, total_oversized);
    std::cout << "Oversized part count: " << total_oversized << std::endl;
}

vtx_vt fix_oversized(const problem& prob, vtx_vt part, mem_t& mem, wgt_view_t vtx_w, wgt_view_t cluster_size, gain_t upper_bound, vtx_vt constraint) {
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    vtx_vt oversized_idx = mem.s_mem.vtx1;
    vtx_vt moves = mem.s_mem.vtx2;
    vtx_vt bid = mem.s_mem.vtx3;
    vtx_vt dead_bit = mem.p_mem.lock_bit;
    ordinal_t total_oversized = 0;
    Kokkos::parallel_scan("compute oversized idx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(cluster_size(i) > upper_bound && dead_bit(i) == 0){
            if(final){
                oversized_idx(i) = update;
            }
            update++;
        } else if(final){
            oversized_idx(i) = -1;
        }
    }, total_oversized);
    if(total_oversized == 0){
        vtx_vt only_moves = Kokkos::subview(moves, std::make_pair(0, 0));
        return only_moves;
    }
    ordinal_t sections = max_sections;
    ordinal_t t_minibuckets = max_buckets*total_oversized*sections;
    if(t_minibuckets > n) {
        sections = n / (max_buckets*total_oversized);
        if(sections == 0) sections = 1;
        t_minibuckets = max_buckets*total_oversized*sections;
    }
    gain_vt pvals = mem.p_mem.pvals;
    gain_vt bucket_offsets = Kokkos::subview(mem.s_mem.gain1, std::make_pair(static_cast<ordinal_t>(0), t_minibuckets));
    Kokkos::deep_copy(exec_space(), bucket_offsets, 0);
    gain_vt vscore = mem.s_mem.gain2;
    Kokkos::parallel_for("compute scores", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t idx = oversized_idx(part(i));
        if(idx == -1) return;
        // make this the full objective?
        gain_t gain = -pvals(i);
        ordinal_t gain_type = gain_bucket(gain, vtx_w(i));
        ordinal_t g_id = (max_buckets*idx + gain_type) * sections + (i % sections);
        bid(i) = g_id;
        vscore(i) = Kokkos::atomic_fetch_add(&bucket_offsets(g_id), vtx_w(i));
    });
    Kokkos::parallel_for("scan score buckets", team_policy_t(1, 1024), KOKKOS_LAMBDA(const member& t){
        //this scan is small so do it within a team instead of an entire grid to save kernel launch time
        Kokkos::parallel_scan(Kokkos::TeamThreadRange(t, 0, t_minibuckets), [&] (const ordinal_t i, gain_t& update, const bool final) {
            gain_t x = bucket_offsets(i);
            if(final){
                bucket_offsets(i) = update;
            }
            update += x;
        });
    });
    ordinal_t width = max_buckets*sections;
    ordinal_t num_moves = 0;
    Kokkos::parallel_scan("filter scores below limit", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t b = bid(i);
        ordinal_t p = part(i);
        ordinal_t idx = oversized_idx(p);
        if(idx == -1) return;
        ordinal_t begin_bucket = idx*width;
        gain_t score = vscore(i) + bucket_offsets(b) - bucket_offsets(begin_bucket);
        gain_t limit = cluster_size(p) - upper_bound;
        if(score < limit){
            if(final){
                moves(update) = i;
                vscore(i) = score;
            }
            update++;
        }
    }, num_moves);
    vtx_vt only_moves = Kokkos::subview(moves, std::make_pair(static_cast<ordinal_t>(0), num_moves));
    vtx_vt dest_part = mem.p_mem.dest_part;
    gain_svt cluster_count = mem.s_mem.max_vwgt;
    // compute number of new clusters needed
    // evicted vertices are sent to "overflow" clusters created for each oversized cluster
    Kokkos::parallel_scan("compute oversized idx", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(cluster_size(i) > upper_bound && dead_bit(i) == 0){
            if(final){
                oversized_idx(i) = update;
            }
            ordinal_t diff = cluster_size(i) - upper_bound;
            // round up
            update += (diff + upper_bound - 1) / upper_bound;
        } else if(final){
            oversized_idx(i) = -1;
        }
    }, cluster_count);
    vtx_vt new_clusters = mem.s_mem.vtx3;
    // identify unused cluster ids
    ordinal_t empty_clusters = 0;
    Kokkos::parallel_scan("compute destinations", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        if(cluster_size(i) == 0){
            if(final){
                new_clusters(update) = i;
            }
            update++;
        }
    }, empty_clusters);
    Kokkos::parallel_for("assign new part", policy_t(0, num_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t v = only_moves(x);
        ordinal_t idx = oversized_idx(part(v));
        ordinal_t offset = vscore(v) / upper_bound;
        // determine if a vertex overflows the overflow cluster
        if((vscore(v) + vtx_w(v) - 1) / upper_bound == offset){
            ordinal_t read = idx + offset;
            dest_part(v) = new_clusters(read);
        } else {
            //assign such vertices to a singleton cluster
            ordinal_t read = Kokkos::atomic_fetch_add(&cluster_count(), 1);
            // this can happen in the case of false overflow that we can't detect properly
            if(read < empty_clusters) dest_part(v) = new_clusters(read);
            else dest_part(v) = new_clusters(idx + offset);
        }
        ordinal_t dest = dest_part(v);
        constraint(dest) = constraint(part(v));
    });
    return only_moves;
}

// stream-compacts order2 for the vertices adjacent to any changed vertex
vtx_vt find_affected(const problem& prob, const vtx_vt swaps, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    vtx_vt swap_bit = mem.s_mem.zeros1;
    ordinal_t total = 0;
    vtx_vt vtx1 = mem.s_mem.vtx1;
    vtx_vt order1 = mem.p_mem.order1;
    vtx_vt order2 = mem.p_mem.order2;
    vtx_vt dest_cache = mem.p_mem.dest_part;
    Kokkos::parallel_for("mark", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        swap_bit(i) = 1;
    });
    // usually, but not always, faster than directly marking adjacencies of moved vertices, since we can break out of the loop
    // also can't use a team here because we want to be able to exit the loop early
    Kokkos::parallel_for("check adjacent", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        if(swap_bit(i) == 1) return;
        //mark adjacent vertices
        edge_offset_t limit = g.graph.row_map(i) + LARGE_CUTOFF;
        if(g.graph.row_map(i+1) < limit) limit = g.graph.row_map(i+1);
        for(edge_offset_t j = g.graph.row_map(i); j < limit; j++){
            ordinal_t v = g.graph.entries(j);
            if(swap_bit(v) == 1){
                swap_bit(i) = 2;
                break;
            }
        }
    });
    ordinal_t bigger_begin = mem.p_mem.offset_large;
    Kokkos::parallel_scan("collect vtx to be checked", policy_t(bigger_begin, g.numRows()), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update, const bool final){
        ordinal_t i = order1(x);
        if(swap_bit(i) == 0){
            if(final){
                vtx1(update) = i;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    // above scan won't run if bigger_begin == g.numRows(), so scan_host won't be set
    if(bigger_begin == g.numRows()){
        total = 0;
    } else {
        exec_space().fence();
        total = mem.s_mem.scan_host();
    }
    Kokkos::parallel_for("check adjacent (large rows)", team_policy_t(total, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        //mark adjacent vertices
        ordinal_t marked = 0;
        ordinal_t i = vtx1(t.league_rank());
        // we have already checked [g.graph.row_map(i), g.graph.row_map(i) + LARGE_CUTOFF)
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i) + LARGE_CUTOFF, g.graph.row_map(i+1)), [=](const edge_offset_t j, ordinal_t& update){
            if(update == 0){
                ordinal_t v = g.graph.entries(j);
                if(swap_bit(v) == 1){
                    update++;
                }
            }
        }, marked);
        if(marked > 0){
            swap_bit(i) = 2;
        }
    });
    vtx_pin_st pin_host = mem.s_mem.pin_host;
    ordinal_t big_begin = mem.p_mem.offset_mid;
    // order2 is already organized into two buckets by degree > or <= 32
    Kokkos::parallel_scan("collect vtx to be updated", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t x, ordinal_t& update, const bool final){
        if(final && x == big_begin){
            pin_host() = update;
        }
        ordinal_t i = order2(x);
        if(swap_bit(i)){
            if(final){
                vtx1(update) = i;
                // reset to zero for next iteration
                swap_bit(i) = 0;
                dest_cache(i) = NULL_PART;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    return vtx1;
}

// updates datastructures assuming a "large" number of vertices are moved
template <bool uniform>
void update_large(const problem& prob, const vtx_vt part, const vtx_vt swaps, cdata_t& cdata, mem_t& mem){
    const matrix_t& g = prob.g;
    vtx_vt vtx1 = find_affected(prob, swaps, mem);
    ordinal_t total = mem.s_mem.scan_host();
    ordinal_t big_begin = mem.p_mem.offset_mid;
    if(big_begin < g.numRows()){
        big_begin = mem.s_mem.pin_host();
    } else {
        big_begin = total;
    }
    ordinal_t small = big_begin;
    ordinal_t big = total - small;
    // this logic is entangled with find_affected()
    vtx_vt big_rows = Kokkos::subview(vtx1, std::make_pair(big_begin, total));
    vtx_vt small_rows = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), small));
    gain_vt pvals = mem.p_mem.pvals;
    int max_size = 512;
    //recompute conn tables for each vertex adjacent to a moved vertex
    Kokkos::parallel_for("update large (big rows)", team_policy_t(big, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(max_size*sizeof(gain_t) + max_size*sizeof(ordinal_t))), KOKKOS_LAMBDA(const member& t){
        const ordinal_t i = big_rows(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        ordinal_t size = g_end - g_start;
        ordinal_t* s_conn_entries;
        gain_t* s_conn_vals;
        if(size < max_size){
            s_conn_entries = (ordinal_t*) t.team_shmem().get_shmem(sizeof(ordinal_t) * size);
            s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
        } else {
            s_conn_entries = cdata.conn_entries.data() + g_start;
            s_conn_vals = cdata.conn_vals.data() + g_start;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t& j) {
            s_conn_entries[j] = NULL_PART;
            s_conn_vals[j] = 0;
        });
        ordinal_t p_i = part(i);
        t.team_barrier();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            ordinal_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            ordinal_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                ordinal_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o++;
                    p_o = (p_o == size) ? 0 : p_o;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                    if(s_conn_entries[p_o] == p){
                        success = true;
                    } else {
                        p_o++;
                        p_o = (p_o == size) ? 0 : p_o;
                    }
                }
            }
            Kokkos::atomic_add(s_conn_vals + p_o, wgt);
        }, pvals(i));
        if(size < max_size){
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_start, g_end), [&] (const edge_offset_t& j) {
                cdata.conn_entries(j) = s_conn_entries[j - g_start];
                cdata.conn_vals(j) = s_conn_vals[j - g_start];
            });
        }
    });
    Kokkos::parallel_for("update large (small rows)", policy_t(0, small), KOKKOS_LAMBDA(const ordinal_t x){
        const ordinal_t i = small_rows(x);
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        for(edge_offset_t j = g_start; j < g_end; j++) {
            if(cdata.conn_entries(j) != NULL_PART){
                cdata.conn_entries(j) = NULL_PART;
                cdata.conn_vals(j) = 0;
            }
        }
        ordinal_t size = g_end - g_start;
        ordinal_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        ordinal_t p_i = part(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            ordinal_t p = part(v);
            if(p == p_i){
                update += wgt;
                continue;
            }
            while(j + 1 < g.graph.row_map(i+1) && p == part(g.graph.entries(j+1))){
                j++;
                if constexpr(uniform) wgt += 1;
                else wgt += g.values(j);
            }
            ordinal_t p_o = hash(p) % static_cast<uint32_t>(size);
            ordinal_t px = s_conn_entries[p_o];
            while(px != p && px != NULL_PART){
                p_o++;
                p_o = (p_o == size) ? 0 : p_o;
                px = s_conn_entries[p_o];
            }
            if(px != p){
                s_conn_entries[p_o] = p;
            }
            s_conn_vals[p_o] += wgt;
        }
        pvals(i) = update;
    });
}

//update datastructures assuming a "small" number of vertices are moved
//2 kernels, 0 device-host syncs
template <bool uniform>
void update_small(const problem& prob, const vtx_vt part, const vtx_vt swaps, const vtx_vt dest_part, cdata_t& cdata, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    gain_vt pvals = mem.p_mem.pvals;
    Kokkos::parallel_for("update small (subtract)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        ordinal_t p = part(i);
        //subtract i's contribution to p connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            if(p == part(v)){
                Kokkos::atomic_add(&pvals(v), -wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            ordinal_t v_size = cdata.conn_table_sizes(v);
            ordinal_t p_o = hash(p) % static_cast<uint32_t>(v_size);
            //v is always adjacent to p because it is adjacent to i which is in p
            while(cdata.conn_entries(v_start + p_o) != p){
                p_o++;
                p_o = (p_o == v_size) ? 0 : p_o;
            }
            //DO NOT USE ATOMIC_ADD_FETCH HERE IT IS WAY SLOWER
            gain_t x = Kokkos::atomic_fetch_add(&cdata.conn_vals(v_start + p_o), -wgt);
            //parts have locked locations if v_size == k (even when not originally allocated to size k)
            if(x == wgt){
                //free this gain slot
                cdata.conn_entries(v_start + p_o) = HASH_RECLAIM;
            }
        });
    });

    // remove the new cluster id from hashmap, and insert old cluster id into hashmap
    Kokkos::parallel_for("swap pval and bval", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = swaps(x);
        // remove best (new cluster) from the hashmap
        ordinal_t best = dest_part(i);
        // insert p (old cluster) into the hashmap
        ordinal_t p = part(i);
        if(best == p) return;
        part(i) = best;
        dest_part(i) = NULL_PART;
        ordinal_t size = cdata.conn_table_sizes(i);
        edge_offset_t offset = cdata.conn_offsets(i);
        ordinal_t b_hash = hash(best) % static_cast<uint32_t>(size);
        gain_t old_val = pvals(i);
        pvals(i) = 0;
        // find new cluster's connection strength, and set into pval
        for(ordinal_t q = 0; q < size; q++){
            ordinal_t p_i = (b_hash + q) % size;
            if(cdata.conn_entries(offset + p_i) == best){
                pvals(i) = cdata.conn_vals(offset + p_i);
                // delete from hashmap
                cdata.conn_vals(offset + p_i) = 0;
                cdata.conn_entries(offset + p_i) = HASH_RECLAIM;
                break;
            } else if(cdata.conn_entries(offset + p_i) == NULL_PART){
                break;
            }
        }
        // DO NOT insert old cluster into hashmap if it has no connection to vertex
        if(old_val == 0) return;
        bool success = false;
        ordinal_t p_o = hash(p) % static_cast<uint32_t>(size);
        // insert p into conn table
        // needs to find either HASH_RECLAIM or NULL_PART to make insertion
        while(!success){
            ordinal_t px = cdata.conn_entries(offset + p_o);
            while(px > NULL_PART){
                p_o = (p_o + 1) % size;
                px = cdata.conn_entries(offset + p_o);
            }
            if(px <= NULL_PART) {
                cdata.conn_entries(offset + p_o) = p;
                success = true;
            }
        }
        cdata.conn_vals(offset + p_o) = old_val;
    });

    Kokkos::parallel_for("update small (add)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        //part contains new part at this point
        ordinal_t best = part(i);
        //add i's contribution to best connectivity for adjacent vertices
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            dest_part(v) = NULL_PART;
            if(best == part(v)){
                Kokkos::atomic_add(&pvals(v), wgt);
                return;
            }
            edge_offset_t v_start = cdata.conn_offsets(v);
            ordinal_t v_size = cdata.conn_table_sizes(v);
            ordinal_t p_o = hash(best) % static_cast<uint32_t>(v_size);
            bool success = false;
            //check if best in conn table
            //can only determine best is absent if NULL_PART is found or v_size reached
            for(ordinal_t q = 0; q < v_size; q++){
                ordinal_t p_i = (p_o + q) % v_size;
                ordinal_t px = cdata.conn_entries(v_start + p_i);
                if(px == best){
                    success = true;
                    p_o = p_i;
                    break;
                } else if(px == NULL_PART){
                    break;
                }
            }
            //insert best into conn table
            //needs to find either HASH_RECLAIM or NULL_PART to make insertion
            while(!success){
                ordinal_t px = cdata.conn_entries(v_start + p_o);
                while(px != best && px > NULL_PART){
                    p_o++;
                    p_o = (p_o == v_size) ? 0 : p_o;
                    px = cdata.conn_entries(v_start + p_o);
                }
                if(px == best){
                    success = true;
                } else {
                    ordinal_t orig = NULL_PART;
                    if(cdata.conn_entries(v_start + p_o) == HASH_RECLAIM) orig = HASH_RECLAIM;
                    //don't care if this thread succeeds if another thread succeeds with the same value
                    Kokkos::atomic_compare_exchange(&cdata.conn_entries(v_start + p_o), orig, best);
                    if(cdata.conn_entries(v_start + p_o) == best){
                        success = true;
                    } else {
                        p_o++;
                        p_o = (p_o == v_size) ? 0 : p_o;
                    }
                }
            }
            Kokkos::atomic_add(&cdata.conn_vals(v_start + p_o), wgt);
        });
    });
}

KOKKOS_INLINE_FUNCTION
static gain_t lookup(const ordinal_t* keys, const gain_t* vals, const ordinal_t& target, const ordinal_t& size){
    ordinal_t start = hash(target) % static_cast<uint32_t>(size);
    for(ordinal_t q = 0; q < size; q++){
        ordinal_t p_i = (start + q) % size;
        if(keys[p_i] == target){
            return vals[p_i];
        } else if(keys[p_i] == NULL_PART){
            return 0;
        }
    }
    return 0;
}

gain_t pval_sum(gain_vt pvals, ordinal_t n){
    // this works well for large vertex swap counts
    // perhaps the old approach could be useful for small vertex swap counts (specifically during the uncoarsening pass)
    gain_t sum = 0;
    Kokkos::parallel_reduce("count cutsize change part1", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i, gain_t& gain_update){
        gain_update += pvals(i);
    }, sum);
    return sum;
}

//perform swaps, update gains, and compute change to cut and imbalance
//4 kernels, 1 device-host syncs
template <bool uniform>
void perform_moves(const problem& prob, vtx_vt part, const vtx_vt swaps, cdata_t& cdata, mem_t& mem, refine_data& curr_state, wgt_view_t vtx_w){
    const wgt_view_t& wdeg = prob.wdeg;
    vtx_vt dest_part = Kokkos::subview(mem.p_mem.dest_part, std::make_pair(static_cast<ordinal_t>(0), prob.g.numRows()));
    ordinal_t total_moves = swaps.extent(0);
    gain_vt pvals = mem.p_mem.pvals;
    wgt_view_t total_deg = curr_state.total_deg;
    wgt_view_t cluster_size = curr_state.cluster_size;
    Kokkos::parallel_for("update total deg", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x){
        ordinal_t i = swaps(x);
        ordinal_t best = dest_part(i);
        ordinal_t p = part(i);
        Kokkos::atomic_add(&cluster_size(best), vtx_w(i));
        Kokkos::atomic_add(&total_deg(p), -wdeg(i));
        Kokkos::atomic_add(&total_deg(best), wdeg(i));
        Kokkos::atomic_add(&cluster_size(p), -vtx_w(i));
    });
    //change part assignments and update part sizes
    if(!cdata.init || total_moves >= prob.g.numRows() * 0.04){
        // update cluster ids before updating datastructures
        Kokkos::parallel_for("update parts", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = swaps(x);
            ordinal_t best = dest_part(i);
            part(i) = best;
        });
        if(!cdata.init){
            init_conn_graph<uniform>(prob, part, cdata, mem);
            Kokkos::deep_copy(exec_space(), dest_part, NULL_PART);
        } else {
            update_large<uniform>(prob, part, swaps, cdata, mem);
        }
    } else {
        // cluster ids updated inside this function
        update_small<uniform>(prob, part, swaps, dest_part, cdata, mem);
    }
    gain_t curr_pval = pval_sum(pvals, prob.g.numRows());
    gain_t cut_change = curr_pval - curr_state.last_pval;
    curr_state.last_pval = curr_pval;
    curr_state.cut -= cut_change;
    curr_state.obj = curr_state.objective();
}

void fast_fill(vtx_vt a, ordinal_t V){
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

//initialize conn hash tables for each vertex
template <bool uniform>
void init_conn_graph(const problem& prob, const vtx_vt& part, cdata_t& cdata, mem_t& mem){
    const matrix_t& g = prob.g;
    cdata.init = true;
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    fast_fill(cdata.conn_entries, NULL_PART);
    // Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    ordinal_t n = g.numRows();
    vtx_vt order2 = mem.p_mem.order2;
    gain_vt pvals = mem.p_mem.pvals;
    vtx_vt big = Kokkos::subview(order2, std::make_pair(mem.p_mem.offset_mid, n));
    vtx_vt small = Kokkos::subview(order2, std::make_pair(static_cast<ordinal_t>(0), mem.p_mem.offset_mid));
    int max_size = 512;
    Kokkos::parallel_for("init conn DS", team_policy_t(n - mem.p_mem.offset_mid, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(max_size*sizeof(gain_t) + max_size*sizeof(ordinal_t))), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = big(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        ordinal_t size = g_end - g_start;
        ordinal_t* s_conn_entries;
        gain_t* s_conn_vals;
        if(size < max_size){
            s_conn_entries = (ordinal_t*) t.team_shmem().get_shmem(sizeof(ordinal_t) * size);
            s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t& j) {
                s_conn_entries[j] = NULL_PART;
                s_conn_vals[j] = 0;
            });
            t.team_barrier();
        } else {
            s_conn_entries = cdata.conn_entries.data() + g_start;
            s_conn_vals = cdata.conn_vals.data() + g_start;
        }
        ordinal_t p_i = part(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            ordinal_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            ordinal_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                ordinal_t px = s_conn_entries[p_o];
                while(px != p && px != NULL_PART){
                    p_o++;
                    p_o = (p_o == size) ? 0 : p_o;
                    px = s_conn_entries[p_o];
                }
                if(px == p){
                    success = true;
                } else {
                    Kokkos::atomic_compare_exchange(s_conn_entries + p_o, NULL_PART, p);
                    if(s_conn_entries[p_o] == p){
                        success = true;
                    } else {
                        p_o++;
                        p_o = (p_o == size) ? 0 : p_o;
                    }
                }
            }
            Kokkos::atomic_add(s_conn_vals + p_o, wgt);
        }, pvals(i));
        if(size < max_size){
            Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g_start, g_end), [&] (const edge_offset_t& j) {
                cdata.conn_entries(j) = s_conn_entries[j - g_start];
                cdata.conn_vals(j) = s_conn_vals[j - g_start];
            });
        }
    });
    Kokkos::parallel_for("init conn DS", policy_t(0, mem.p_mem.offset_mid), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = small(x);
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        ordinal_t size = g_end - g_start;
        ordinal_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            ordinal_t p = part(v);
            if(p == part(i)){
                update += wgt;
                continue;
            }
            while(j + 1 < g.graph.row_map(i+1) && p == part(g.graph.entries(j+1))){
                j++;
                if constexpr(uniform) wgt += 1;
                else wgt += g.values(j);
            }
            ordinal_t p_o = hash(p) % static_cast<uint32_t>(size);
            ordinal_t px = s_conn_entries[p_o];
            while(px != p && px != NULL_PART){
                p_o++;
                p_o = (p_o == size) ? 0 : p_o;
                px = s_conn_entries[p_o];
            }
            if(px != p){
                s_conn_entries[p_o] = p;
            }
            s_conn_vals[p_o] += wgt;
        }
        pvals(i) = update;
    });
}

//initializes datastructures
cdata_t truncate_and_init_mem(mem_t& mem, problem& prob, int label_count, bool top){
    const matrix_t g = prob.g;
    ordinal_t n = g.numRows();
    cdata_t cdata;
    cdata.init = false;
    cdata.conn_offsets = Kokkos::subview(mem.p_mem.row_map, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    cdata.conn_table_sizes = Kokkos::subview(mem.p_mem.cluster_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(!top) degree *= 1.2;
        if(degree > label_count) degree = label_count;
        cdata.conn_offsets(i) = degree;
        cdata.conn_table_sizes(i) = degree;
    });
    // rather than organizing vertices into 3 buckets
    // organize vertices into two different sets of two buckets each
    // I found that the performance of using 3 buckets was worse (likely due to poorer cache utilization)
    vtx_vt order1 = mem.p_mem.order1;
    Kokkos::parallel_scan("generate order1", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree < LARGE_CUTOFF){
            if(final){
                order1(update) = i;
            }
            update++;
        } else if(final){
            order1(n - 1 - (i - update)) = i;
        }
    }, mem.p_mem.offset_large);
    vtx_vt order2 = mem.p_mem.order2;
    Kokkos::parallel_scan("generate order2", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree < MID_CUTOFF){
            if(final){
                order2(update) = i;
            }
            update++;
        } else if(final){
            order2(n - 1 - (i - update)) = i;
        }
    }, mem.p_mem.offset_mid);
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        edge_offset_t x = cdata.conn_offsets(i);
        if(final){
            cdata.conn_offsets(i) = update;
        }
        update += x;
        if(final && i + 1 == n){
            cdata.conn_offsets(n) = update;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    gain_size = mem.s_mem.scan_host();
    cdata.conn_vals = Kokkos::subview(mem.p_mem.vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.conn_entries = Kokkos::subview(mem.p_mem.entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.c_graph = matrix_t("conn graph", g.numRows(), g.numRows(), gain_size, cdata.conn_vals, cdata.conn_offsets, cdata.conn_entries);
    gain_vt pval_init_subview = Kokkos::subview(mem.p_mem.pvals, std::make_pair(static_cast<ordinal_t>(0), n));
    vtx_vt dest_part_init_subview = Kokkos::subview(mem.p_mem.dest_part, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::deep_copy(exec_space(), pval_init_subview, 0);
    Kokkos::deep_copy(exec_space(), dest_part_init_subview, NULL_PART);
    return cdata;
}

void clone_pval(mem_t& mem, ordinal_t n){
    gain_vt pval_subview = Kokkos::subview(mem.p_mem.pvals, std::make_pair(static_cast<ordinal_t>(0), n));
    gain_vt pval_clone_subview = Kokkos::subview(mem.p_mem.pvals_clone, std::make_pair(static_cast<ordinal_t>(0), n));
    Kokkos::deep_copy(exec_space(), pval_clone_subview, pval_subview);
}

template <bool uniform>
void jet_refine(const matrix_t g, wgt_view_t wdeg, wgt_view_t vtx_w, vtx_vt best_part, refine_data& best_state, bool is_initial, gain_t upper_bound, mem_t& mem, const vtx_vt constraint){
    // vertices that are oversized before clustering can not join any clusters, nor can their cluster be joined
    Kokkos::parallel_for("lock overwgt", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        if(vtx_w(i) >= upper_bound) mem.p_mem.lock_bit(i) = 1;
        else mem.p_mem.lock_bit(i) = 0;
    });
    problem prob;
    prob.g = g;
    prob.wdeg = wdeg;
    prob.use_team = (g.nnz() / g.numRows() >= 8);
    refine_data curr_state(best_state);
    vtx_vt part = Kokkos::subview(mem.p_mem.part, std::make_pair(static_cast<ordinal_t>(0), g.numRows()));
    Kokkos::deep_copy(exec_space(), part, best_part);
    vtx_vt c_constraint("cluster constraint", g.numRows());
    Kokkos::deep_copy(exec_space(), c_constraint, constraint);
    cdata_t cdata = truncate_and_init_mem(mem, prob, best_state.label_count, best_state.g_deg == g.nnz());
    if(!is_initial){
        init_conn_graph<uniform>(prob, part, cdata, mem);
        curr_state.last_pval = pval_sum(mem.p_mem.pvals, g.numRows());
    } else {
        curr_state.last_pval = 0;
    }
    int iter_count = 0;
    Kokkos::fence();
    Kokkos::Timer iter_t;
    std::vector<float> filter_ratios = {0.75, 0.25};
    std::vector<int> limits = {4, 2};
    for(size_t x = 0; x < filter_ratios.size(); x++){
        float filter_ratio = filter_ratios[x];
        int limit = limits[x];
        int count = 0;
        while(count++ < limit){
            iter_count++;
            vtx_vt moves;
            matrix_t c_graph = cdata.c_graph;
            if(!cdata.init){
                // use the input graph in place of the conn graph
                c_graph = g;
            }
            moves = jet_lp<uniform>(prob, c_graph, part, curr_state, mem, filter_ratio, constraint, c_constraint);
            if(moves.extent(0) == 0) break;
            perform_moves<uniform>(prob, part, moves, cdata, mem, curr_state, vtx_w);
            moves = fix_oversized(prob, part, mem, vtx_w, curr_state.cluster_size, upper_bound, c_constraint);
            if(moves.extent(0) > 0){
                perform_moves<uniform>(prob, part, moves, cdata, mem, curr_state, vtx_w);
                // count_oversized(g.numRows(), cluster_size, upper_bound);
            }
            //copy current partition and relevant data to output partition if following conditions pass
            if(curr_state.obj > best_state.obj){
                best_state.copy(curr_state);
                Kokkos::deep_copy(exec_space(), best_part, part);
                clone_pval(mem, g.numRows());
            }
        }
    }
    Kokkos::fence();
    relabel_contiguously(best_part, best_state, mem);
}
};

}
