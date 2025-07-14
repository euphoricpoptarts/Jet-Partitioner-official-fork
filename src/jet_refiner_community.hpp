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
#include "memory_store_community.hpp"
#include "part_stat_community.hpp"

namespace jet_community {

template<class crsMat, typename part_t>
class jet_refiner {
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
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using vtx_svt = Kokkos::View<ordinal_t, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using obj_vt = Kokkos::View<float*, Device>;
    using edge_subview_t = Kokkos::View<edge_offset_t, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using dyn_policy_t = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using dyn_team_policy_t = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>, exec_space>;
    using member = typename team_policy_t::member_type;
    using stat = jet_community::part_stat<matrix_t, part_t>;
    using refine_data = typename stat::refine_data;
    using mem_t = jet_community::memory_store<matrix_t, part_t>;
    using cdata_t = typename mem_t::conn_data;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr float OBJ_MIN = std::numeric_limits<float>::lowest();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr part_t NULL_PART = -1;
    static constexpr part_t HASH_RECLAIM = -2;
    static constexpr part_t NO_MOVE = -3;
    static constexpr ordinal_t MID_CUTOFF = 32;
    static constexpr ordinal_t LARGE_CUTOFF = 128;

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

void copy_refine_data(refine_data& lhs, refine_data& rhs){
    Kokkos::deep_copy(exec_space(), lhs.total_deg, rhs.total_deg);
    lhs.g_deg = rhs.g_deg;
    lhs.cut = rhs.cut;
    lhs.init = rhs.init;
    lhs.mod = rhs.mod;
    lhs.label_count = rhs.label_count;
}

refine_data clone_refine_data(refine_data& rhs){
    refine_data clone;
    clone.total_deg = gain_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
    copy_refine_data(clone, rhs);
    return clone;
}

void relabel_contiguously(part_vt labels, refine_data& rfd, mem_t& mem){
	ordinal_t n = labels.extent(0);
    ordinal_t initial_count = rfd.label_count;
	vtx_view_t used = Kokkos::subview(mem.s_mem.vtx1, std::make_pair((ordinal_t)0, initial_count));
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
    gain_vt total_deg("new total degree", t_labels);
    Kokkos::parallel_for("relabel degrees", policy_t(0, initial_count), KOKKOS_LAMBDA(const ordinal_t i){
		if(rfd.total_deg(i) > 0){
            ordinal_t relabeled = used(i);
            total_deg(relabeled) = rfd.total_deg(i);
        }
	});
    rfd.total_deg = total_deg;
    rfd.label_count = t_labels;
}

//determines which vertices (if any) should be moved to another part to improve objective
//8 kernels, 2 device-host syncs
template <bool uniform>
vtx_view_t jet_lp(const problem& prob, const matrix_t& c_graph, const part_vt& part, const refine_data& rfd, mem_t& mem, float filter_ratio){
    const matrix_t& g = prob.g;
    ordinal_t n = g.numRows();
    ordinal_t num_pos = 0;
    part_vt dest_part = mem.p_mem.dest_part;
    obj_vt save_gains = mem.p_mem.gain_persistent;
    vtx_view_t swap_bit = mem.s_mem.zeros1;
    gain_vt total_deg = rfd.total_deg;
    gain_vt wdeg = prob.wdeg;
    cdata_t& cdata = mem.cd_mem;
    part_vt conn_table_sizes = cdata.conn_table_sizes;
    gain_vt pvals = mem.p_mem.pvals;
    float inv_2m = stat::get_penalty() / static_cast<float>(rfd.g_deg);
    vtx_view_t vtx1 = mem.s_mem.vtx1;
    vtx_view_t vtx2 = mem.s_mem.vtx2;
    vtx_view_t order1 = mem.p_mem.order1;
    ordinal_t big_begin = mem.p_mem.offset_large;
    vtx_view_t small_tables = Kokkos::subview(order1, std::make_pair(static_cast<ordinal_t>(0), big_begin));
    vtx_view_t large_tables = Kokkos::subview(order1, std::make_pair(big_begin, n));
    // if input label count is smaller than LARGE_CUTOFF, then all tables are also smaller than LARGE_CUTOFF
    bool truncated = (rfd.label_count <= LARGE_CUTOFF);
    Kokkos::parallel_for("select destination part (small tables)", policy_t(0, truncated ? n : big_begin), KOKKOS_LAMBDA(const ordinal_t x){
        ordinal_t i = truncated ? x : small_tables(x);
        if(dest_part(i) != NULL_PART){
            return;
        }
        part_t best = NO_MOVE;
        float wd = wdeg(i);
        float multi = wd*inv_2m;
        part_t p = part(i);
        float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
        // b_conn must be at least this value to pass filter
        float b_conn = p_conn - filter_ratio*(p_conn);
        edge_offset_t start = c_graph.graph.row_map(i);
        edge_offset_t end = c_graph.graph.row_map(i+1);
        //finds potential destination as most connected part excluding p
        for(edge_offset_t j = start; j < end; j++){
            gain_t j_val = c_graph.values(j);
            if(j_val > 0 && j_val >= b_conn){
                part_t px = c_graph.graph.entries(j);
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
            float multi = wd*inv_2m;
            edge_offset_t start = c_graph.graph.row_map(i);
            edge_offset_t end = c_graph.graph.row_map(i+1);
            part_t p = part(i);
            float p_conn = pvals(i) - (total_deg(p) - wd)*multi;
            // j_conn must be at least this value to pass filter
            float maxl = p_conn - filter_ratio*(p_conn);
            part_t argmax = NO_MOVE;
            //finds potential destination as most connected part excluding p
            for(edge_offset_t j = start + t.team_rank(); j < end; j += team_size){
                gain_t j_val = c_graph.values(j);
                if(j_val > 0 && j_val >= maxl){
                    part_t px = c_graph.graph.entries(j);
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
            part_t argmaxg = NO_MOVE;
            t.team_reduce(Kokkos::Min<part_t, mem_space>(argmaxg), argmax);
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
        part_t best = dest_part(i);
        if(best != NO_MOVE){
            if(final){
                vtx1(update) = i;
            }
            update++;
        }
    }, mem.s_mem.scan_host);
    exec_space().fence();
    num_pos = mem.s_mem.scan_host();
    //truncate scratch views by num_pos
    vtx_view_t pos_moves = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), num_pos));
    if(big_begin < n){
        big_begin = pin_host();
    } else {
        big_begin = num_pos;
    }
    ordinal_t small = big_begin;
    ordinal_t big = num_pos - small;
    vtx_view_t big_rows = Kokkos::subview(vtx1, std::make_pair(big_begin, num_pos));
    vtx_view_t small_rows = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), small));
    float eps = 0.1;
    //in this kernel every potential move from the previous filters
    //is reevaluated by considering the effect of the other potential moves
    //a move is considered to occur before another according to their potential gains
    //and the vertex ids
    Kokkos::parallel_for("afterburner heuristic", team_policy_t(big, 256), KOKKOS_LAMBDA(const member& t){
        float change = 0;
        ordinal_t i = big_rows(t.league_rank());
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&](const edge_offset_t j, float& update){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
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
        part_t best = dest_part(i);
        part_t p = part(i);
        float wd = prob.wdeg(i);
        float multi = wd*inv_2m;
        float igain = pregain(i);
        ordinal_t hi = hash(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            float vgain = pregain(v);
            //adjust local gain if v has higher priority than i
            if((vgain - igain) >= eps || (abs(vgain - igain) < eps && static_cast<ordinal_t>(hash(v)) < hi)){
                part_t vpart = dest_part(v);
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
    vtx_view_t swaps2 = Kokkos::subview(vtx2, std::make_pair(static_cast<ordinal_t>(0), num_pos));
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

// stream-compacts order2 for the vertices adjacent to any changed vertex
vtx_view_t find_affected(const problem& prob, const vtx_view_t swaps, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    vtx_view_t swap_bit = mem.s_mem.zeros1;
    ordinal_t total = 0;
    vtx_view_t vtx1 = mem.s_mem.vtx1;
    vtx_view_t order1 = mem.p_mem.order1;
    vtx_view_t order2 = mem.p_mem.order2;
    vtx_view_t dest_cache = mem.p_mem.dest_part;
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
void update_large(const problem& prob, const part_vt part, const vtx_view_t swaps, mem_t& mem){
    const matrix_t& g = prob.g;
    cdata_t& cdata = mem.cd_mem;
    vtx_view_t vtx1 = find_affected(prob, swaps, mem);
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
    vtx_view_t big_rows = Kokkos::subview(vtx1, std::make_pair(big_begin, total));
    vtx_view_t small_rows = Kokkos::subview(vtx1, std::make_pair(static_cast<ordinal_t>(0), small));
    gain_vt pvals = mem.p_mem.pvals;
    int max_size = 512;
    //recompute conn tables for each vertex adjacent to a moved vertex
    Kokkos::parallel_for("update large (big rows)", team_policy_t(big, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(max_size*sizeof(gain_t) + max_size*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        const ordinal_t i = big_rows(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t* s_conn_entries;
        gain_t* s_conn_vals;
        if(size < max_size){
            s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
            s_conn_vals = (gain_t*) t.team_shmem().get_shmem(sizeof(gain_t) * size);
        } else {
            s_conn_entries = cdata.conn_entries.data() + g_start;
            s_conn_vals = cdata.conn_vals.data() + g_start;
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, size), [&] (const edge_offset_t& j) {
            s_conn_entries[j] = NULL_PART;
            s_conn_vals[j] = 0;
        });
        part_t p_i = part(i);
        t.team_barrier();
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
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
        part_t size = g_end - g_start;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        part_t p_i = part(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                continue;
            }
            while(j + 1 < g.graph.row_map(i+1) && p == part(g.graph.entries(j+1))){
                j++;
                if constexpr(uniform) wgt += 1;
                else wgt += g.values(j);
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            part_t px = s_conn_entries[p_o];
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
void update_small(const problem& prob, const part_vt part, const vtx_view_t swaps, const part_vt dest_part, mem_t& mem){
    const matrix_t& g = prob.g;
    ordinal_t total_moves = swaps.extent(0);
    gain_vt pvals = mem.p_mem.pvals;
    cdata_t& cdata = mem.cd_mem;
    Kokkos::parallel_for("update small (subtract)", team_policy_t(total_moves, Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = swaps(t.league_rank());
        part_t p = part(i);
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
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(p) % static_cast<uint32_t>(v_size);
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
        part_t best = dest_part(i);
        // insert p (old cluster) into the hashmap
        part_t p = part(i);
        part(i) = best;
        dest_part(i) = NULL_PART;
        part_t size = cdata.conn_table_sizes(i);
        edge_offset_t offset = cdata.conn_offsets(i);
        part_t b_hash = hash(best) % static_cast<uint32_t>(size);
        gain_t old_val = pvals(i);
        pvals(i) = 0;
        // find new cluster's connection strength, and set into pval
        for(part_t q = 0; q < size; q++){
            part_t p_i = (b_hash + q) % size;
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
        part_t p_o = hash(p) % static_cast<uint32_t>(size);
        // insert p into conn table
        // needs to find either HASH_RECLAIM or NULL_PART to make insertion
        while(!success){
            part_t px = cdata.conn_entries(offset + p_o);
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
        part_t best = part(i);
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
            part_t v_size = cdata.conn_table_sizes(v);
            part_t p_o = hash(best) % static_cast<uint32_t>(v_size);
            bool success = false;
            //check if best in conn table
            //can only determine best is absent if NULL_PART is found or v_size reached
            for(part_t q = 0; q < v_size; q++){
                part_t p_i = (p_o + q) % v_size;
                part_t px = cdata.conn_entries(v_start + p_i);
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
                part_t px = cdata.conn_entries(v_start + p_o);
                while(px != best && px > NULL_PART){
                    p_o++;
                    p_o = (p_o == v_size) ? 0 : p_o;
                    px = cdata.conn_entries(v_start + p_o);
                }
                if(px == best){
                    success = true;
                } else {
                    part_t orig = NULL_PART;
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
static gain_t lookup(const part_t* keys, const gain_t* vals, const part_t& target, const part_t& size){
    part_t start = hash(target) % static_cast<uint32_t>(size);
    for(part_t q = 0; q < size; q++){
        part_t p_i = (start + q) % size;
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
void perform_moves(const problem& prob, part_vt part, const vtx_view_t swaps, mem_t& mem, refine_data& curr_state){
    const wgt_view_t& wdeg = prob.wdeg;
    vtx_view_t dest_part = mem.p_mem.dest_part;
    ordinal_t total_moves = swaps.extent(0);
    cdata_t& cdata = mem.cd_mem;
    gain_vt pvals = mem.p_mem.pvals;
    Kokkos::parallel_for("update total deg", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t& x){
        ordinal_t i = swaps(x);
        part_t best = dest_part(i);
        part_t p = part(i);
        Kokkos::atomic_add(&curr_state.total_deg(p), -wdeg(i));
        Kokkos::atomic_add(&curr_state.total_deg(best), wdeg(i));
    });
    //change part assignments and update part sizes
    if(!cdata.init || total_moves >= prob.g.numRows() * 0.04){
        // update cluster ids before updating datastructures
        Kokkos::parallel_for("update parts", policy_t(0, total_moves), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = swaps(x);
            part_t best = dest_part(i);
            part(i) = best;
        });
        if(!cdata.init){
            init_conn_graph<uniform>(prob, part, mem);
            Kokkos::deep_copy(exec_space(), dest_part, NULL_PART);
        } else {
            update_large<uniform>(prob, part, swaps, mem);
        }
    } else {
        // cluster ids updated inside this function
        update_small<uniform>(prob, part, swaps, dest_part, mem);
    }
    gain_t curr_pval = pval_sum(pvals, prob.g.numRows());
    int64_t cut_change = curr_pval - curr_state.last_pval;
    curr_state.last_pval = curr_pval;
    curr_state.cut -= cut_change;
    curr_state.mod = stat::modularity(curr_state.g_deg, curr_state.cut, curr_state.total_deg);
}

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

//initialize conn hash tables for each vertex
template <bool uniform>
void init_conn_graph(const problem& prob, const part_vt& part, mem_t& mem){
    const matrix_t& g = prob.g;
    cdata_t& cdata = mem.cd_mem;
    cdata.init = true;
    Kokkos::deep_copy(exec_space(), cdata.conn_vals, 0);
    fast_fill(cdata.conn_entries, NULL_PART);
    // Kokkos::deep_copy(exec_space(), cdata.conn_entries, NULL_PART);
    ordinal_t n = g.numRows();
    vtx_view_t order2 = mem.p_mem.order2;
    gain_vt pvals = mem.p_mem.pvals;
    vtx_view_t big = Kokkos::subview(order2, std::make_pair(mem.p_mem.offset_mid, n));
    vtx_view_t small = Kokkos::subview(order2, std::make_pair(static_cast<ordinal_t>(0), mem.p_mem.offset_mid));
    int max_size = 512;
    Kokkos::parallel_for("init conn DS", team_policy_t(n - mem.p_mem.offset_mid, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(max_size*sizeof(gain_t) + max_size*sizeof(part_t))), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = big(t.league_rank());
        edge_offset_t g_start = cdata.conn_offsets(i);
        edge_offset_t g_end = cdata.conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t* s_conn_entries;
        gain_t* s_conn_vals;
        if(size < max_size){
            s_conn_entries = (part_t*) t.team_shmem().get_shmem(sizeof(part_t) * size);
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
        part_t p_i = part(i);
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [&] (const edge_offset_t& j, gain_t& update){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == p_i){
                update += wgt;
                return;
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            bool success = false;
            while(!success){
                part_t px = s_conn_entries[p_o];
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
        part_t size = g_end - g_start;
        part_t* s_conn_entries = cdata.conn_entries.data() + g_start;
        gain_t* s_conn_vals = cdata.conn_vals.data() + g_start;
        gain_t update = 0;
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt;
            if constexpr(uniform) wgt = 1;
            else wgt = g.values(j);
            part_t p = part(v);
            if(p == part(i)){
                update += wgt;
                continue;
            }
            while(j + 1 < g.graph.row_map(i+1) && p == part(g.graph.entries(j+1))){
                j++;
                if constexpr(uniform) wgt += 1;
                else wgt += g.values(j);
            }
            part_t p_o = hash(p) % static_cast<uint32_t>(size);
            part_t px = s_conn_entries[p_o];
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
void truncate_and_init_mem(mem_t& mem, problem& prob, int label_count, bool top){
    const matrix_t g = prob.g;
    ordinal_t n = g.numRows();
    cdata_t& cdata = mem.cd_mem;
    cdata.init = false;
    cdata.conn_offsets = Kokkos::subview(cdata.conn_offsets, std::make_pair(static_cast<ordinal_t>(0), n + 1));
    cdata.conn_table_sizes = Kokkos::subview(cdata.conn_table_sizes, std::make_pair(static_cast<ordinal_t>(0), n));
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
    vtx_view_t order1 = mem.p_mem.order1;
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
    vtx_view_t order2 = mem.p_mem.order2;
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
    cdata.conn_vals = Kokkos::subview(cdata.conn_vals, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.conn_entries = Kokkos::subview(cdata.conn_entries, std::make_pair(static_cast<edge_offset_t>(0), gain_size));
    cdata.c_graph = matrix_t("conn graph", g.numRows(), g.numRows(), gain_size, cdata.conn_vals, cdata.conn_offsets, cdata.conn_entries);
    Kokkos::deep_copy(exec_space(), mem.p_mem.pvals, 0);
    Kokkos::deep_copy(exec_space(), mem.p_mem.dest_part, NULL_PART);
}

template <bool uniform>
void jet_refine(const matrix_t g, wgt_view_t wdeg, part_vt best_part, refine_data& best_state, bool is_initial, mem_t& input_mem){
    // contains reusable memory to avoid repeated allocations/deallocations
    // some of this memory contains state information
    mem_t mem(input_mem, g);
    // initialize metadata
    if(!best_state.init){
        best_state.mod = -1.0;
        best_state.total_deg = gain_vt("total degree of clusters", g.numRows());
        best_state.g_deg = stat::sum(wdeg);
        best_state.cut = best_state.g_deg;
        best_state.label_count = g.numRows();
        Kokkos::deep_copy(best_state.total_deg, wdeg);
        best_state.init = true;
    }
    problem prob;
    prob.g = g;
    prob.wdeg = wdeg;
    prob.use_team = (g.nnz() / g.numRows() >= 8);
    refine_data curr_state = clone_refine_data(best_state);
    part_vt part = mem.p_mem.part;
    Kokkos::deep_copy(exec_space(), part, best_part);
    truncate_and_init_mem(mem, prob, best_state.label_count, best_state.g_deg == g.nnz());
    if(!is_initial){
        init_conn_graph<uniform>(prob, part, mem);
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
            vtx_view_t moves;
            matrix_t c_graph = mem.cd_mem.c_graph;
            if(!mem.cd_mem.init){
                // use the input graph in place of the conn graph
                c_graph = g;
            }
            moves = jet_lp<uniform>(prob, c_graph, part, curr_state, mem, filter_ratio);
            if(moves.extent(0) == 0) break;
            perform_moves<uniform>(prob, part, moves, mem, curr_state);
            //copy current partition and relevant data to output partition if following conditions pass
            if(curr_state.mod > best_state.mod){
                copy_refine_data(best_state, curr_state);
                Kokkos::deep_copy(exec_space(), best_part, part);
            }
        }
    }
    Kokkos::fence();
    input_mem.p_mem.offset_mid = mem.p_mem.offset_mid;
    input_mem.p_mem.offset_large = mem.p_mem.offset_large;
    relabel_contiguously(best_part, best_state, mem);
}
};

}
