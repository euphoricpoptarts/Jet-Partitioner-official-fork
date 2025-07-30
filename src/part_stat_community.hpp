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

namespace jet_community {

template<class crsMat, typename part_t>
class part_stat {
public:

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    // need some trickery because make_signed is undefined for floating point types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using mem_space = typename matrix_t::memory_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using scalar_t = typename matrix_t::value_type;
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using wgt_view_t = Kokkos::View<scalar_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using gain_2vt = Kokkos::View<gain_t**, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using team_policy_t = Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;
    static constexpr double penalty_scale = 2.0;

// metadata that is preserved between levels in the multilevel scheme
struct refine_data {
    gain_vt total_deg;
    scalar_t g_deg = 0;
    scalar_t v_total = 0;
    gain_t cut = 0;
    // except for this one
    gain_t last_pval = 0;
    double mod = -1.0;
    part_t label_count;
    bool init = false;
};

static gain_t get_total_cut(const matrix_t g, const part_vt partition){
    gain_t total_cut = 0;
    if(!is_host_space && g.nnz() / g.numRows() >= 8){
        Kokkos::parallel_reduce("find total cut (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t, gain_t& update){
            gain_t local_cut = 0;
            ordinal_t i = t.league_rank();
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j, gain_t& local_update){
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_update += wgt;
                }
            }, local_cut);
            Kokkos::single(Kokkos::PerTeam(t), [&] (){
                update += local_cut;
            });
        }, total_cut);
    } else {
        Kokkos::parallel_reduce("find total cut", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
            gain_t local_cut = 0;
            for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                gain_t wgt = g.values(j);
                if(partition(i) != partition(v)){
                    local_cut += wgt;
                }
            }
            update += local_cut;
        }, total_cut);
    }
    return total_cut;
}

static ordinal_t get_total_labels(const part_vt labels){
	ordinal_t n = labels.extent(0);
	vtx_view_t used("used labels", n);
	Kokkos::parallel_for("find used labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		ordinal_t l = labels(i);
		Kokkos::atomic_add(&used(l), 1);
	});
	ordinal_t total = 0;
	Kokkos::parallel_reduce("find used labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update){
		if(used(i) > 0) update++;
	}, total);
	return total;
}

static double get_penalty_modifier(const refine_data& curr_state) {
    double modifier = penalty_scale * static_cast<float>(curr_state.g_deg);
    double inv_vtot = 1.0 / static_cast<float>(curr_state.v_total);
    modifier = modifier * inv_vtot * inv_vtot;
    return modifier;
}

static void relabel(part_vt labels){
	ordinal_t n = labels.extent(0);
	vtx_view_t used("used labels", n);
	Kokkos::parallel_for("find used labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		ordinal_t l = labels(i);
		Kokkos::atomic_add(&used(l), 1);
	});
	Kokkos::parallel_scan("count labels", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
		if(used(i) > 0){
			if(final) used(i) = update;
			update++;
		}
	});
	Kokkos::parallel_for("relabel", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		labels(i) = used(labels(i));
	});
}

static double modularity(refine_data& curr_state){
    double m = 0;
    const gain_vt total = curr_state.total_deg;
    Kokkos::parallel_reduce("sum modularity", policy_t(0, total.size()), KOKKOS_LAMBDA(const ordinal_t l, double& update){
        double total_ratio = static_cast<double>(total(l));
        update -= total_ratio*total_ratio;
    }, m);
    double penalty_factor = 1.0 / static_cast<double>(curr_state.v_total);
    penalty_factor = penalty_scale*penalty_factor*penalty_factor;
    m = m*penalty_factor;
    double inv_gdeg = 1.0 / static_cast<double>(curr_state.g_deg);
    m += 1.0 - static_cast<double>(curr_state.cut) * inv_gdeg;
    // std::cout << "Objective " << m << std::endl;
    return m;
}

static double modularity(const matrix_t g, const part_vt labels, const ordinal_t label_count, scalar_t g_degree){
    wgt_view_t internal("internal degree", label_count);
    wgt_view_t total("total degree", label_count);
    ordinal_t n = g.numRows();
    Kokkos::parallel_for("count degrees", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
		scalar_t id = 0;
        scalar_t td = 0;
        ordinal_t l = labels(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i+1); j++){
            td += g.values(j);
            ordinal_t v = g.graph.entries(j);
            if(l == labels(v)) id += g.values(j);
        }
        Kokkos::atomic_add(&internal(l), id);
        Kokkos::atomic_add(&total(l), td);
	});
    double m = 0;
    Kokkos::parallel_reduce("sum modularity", policy_t(0, label_count), KOKKOS_LAMBDA(const ordinal_t l, double& update){
        double internal_ratio = static_cast<double>(internal(l)) / static_cast<double>(g_degree);
        double total_ratio = static_cast<double>(total(l)) / static_cast<double>(g_degree);
        double l_mod = internal_ratio - penalty_scale*(total_ratio*total_ratio);
        update += l_mod;
    }, m);
    return m;
}

static double modularity(const scalar_t g_degree, const scalar_t cutsize, const gain_vt total){
    double m = 0;
    Kokkos::parallel_reduce("sum modularity", policy_t(0, total.size()), KOKKOS_LAMBDA(const ordinal_t l, double& update){
        double total_ratio = static_cast<double>(total(l));
        update -= total_ratio*total_ratio;
    }, m);
    double inv_gdeg = 1.0 / static_cast<double>(g_degree);
    m = m*penalty_scale*inv_gdeg*inv_gdeg;
    m += 1.0 - static_cast<double>(cutsize) * inv_gdeg;
    return m;
}

static int total_labels(const gain_vt total){
    int labels = 0;
    Kokkos::parallel_reduce("count used labels", policy_t(0, total.size()), KOKKOS_LAMBDA(const ordinal_t i, int& update){
        if(total(i) > 0) update++;
    }, labels);
    return labels;
}

// this is needed in a few different places so it is best to have one implementation for consistency
static ordinal_t optimal_size(const ordinal_t total_size, const part_t k){
    //round up as per convention
    return (total_size + k - 1) / k;
}

static gain_2vt cut_heatmap(const matrix_t g, const part_vt partition, const part_t k){
    gain_2vt heatmap("heatmap", k, k);
    Kokkos::parallel_for("create cut heatmap (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            if(partition(i) != partition(v)){
                Kokkos::atomic_add(&heatmap(partition(i), partition(v)), wgt);
            }
        });
    });
    return heatmap;
}

static gain_vt cut_per_part(const matrix_t g, const part_vt partition, const part_t k){
    gain_vt heatmap("heatmap", k);
    Kokkos::parallel_for("find cut per part (team)", team_policy_t(g.numRows(), Kokkos::AUTO), KOKKOS_LAMBDA(const member& t){
        ordinal_t i = t.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, g.graph.row_map(i), g.graph.row_map(i + 1)), [=] (const edge_offset_t j){
            ordinal_t v = g.graph.entries(j);
            gain_t wgt = g.values(j);
            if(partition(i) != partition(v)){
                Kokkos::atomic_add(&heatmap(partition(i)), wgt);
            }
        });
    });
    return heatmap;
}

static gain_vt get_part_sizes(const matrix_t g, const wgt_view_t vtx_w, const part_vt partition, part_t k){
    gain_vt part_size("part sizes", k);
    Kokkos::parallel_for("calc part sizes", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        part_t p = partition(i);
        Kokkos::atomic_add(&part_size(p), vtx_w(i));
    });
    return part_size;
}

//get sum of vertex weights
static scalar_t get_total_size(const matrix_t g, const wgt_view_t vtx_w){
    scalar_t total_size = 0;
    Kokkos::parallel_reduce("sum of vertex weights", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += vtx_w(i);
    }, total_size);
    return total_size;
}

static gain_t largest_part_size(const gain_vt& ps){
    gain_t result = 0;
    Kokkos::parallel_reduce("get max part size", policy_t(0, ps.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
        if(ps(i) > update){
            update = ps(i);
        }
    }, Kokkos::Max<gain_t, Kokkos::HostSpace>(result));
    return result;
}

static void stash_largest(const gain_vt& ps, gain_svt& result){
    Kokkos::parallel_reduce("get max part size (store in view)", policy_t(0, ps.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, gain_t& update){
        if(ps(i) > update){
            update = ps(i);
        }
    }, Kokkos::Max<gain_t, typename gain_svt::memory_space>(result));
}

static gain_t max_part_cut(const matrix_t g, part_vt part, const part_t k){
    return largest_part_size(cut_per_part(g, part, k));
}

static scalar_t comm_size(const matrix_t& g, const part_vt& part, part_t k){
    ordinal_t n = g.numRows();
    edge_view_t conn_offsets("comp offsets", n + 1);
    Kokkos::parallel_for("comp conn row size", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t& i){
        ordinal_t degree = g.graph.row_map(i + 1) - g.graph.row_map(i);
        if(degree > static_cast<ordinal_t>(k)) degree = k;
        conn_offsets(i + 1) = degree;
    });
    edge_offset_t gain_size = 0;
    Kokkos::parallel_scan("comp conn offsets", policy_t(0, n + 1), KOKKOS_LAMBDA(const ordinal_t& i, edge_offset_t& update, const bool final){
        update += conn_offsets(i);
        if(final){
            conn_offsets(i) = update;
        }
    }, gain_size);
    part_vt conn_entries("conn entries", gain_size);
    part_t NULL_PART = -1;
    Kokkos::deep_copy(exec_space(), conn_entries, NULL_PART);
    scalar_t result = 0;
    Kokkos::parallel_reduce("find communication volume", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t& i, scalar_t& update){
        edge_offset_t g_start = conn_offsets(i);
        edge_offset_t g_end = conn_offsets(i + 1);
        part_t size = g_end - g_start;
        part_t used_cap = 0;
        part_t local = part(i);
        for(edge_offset_t j = g.graph.row_map(i); j < g.graph.row_map(i + 1); j++) {
            ordinal_t v = g.graph.entries(j);
            part_t p = part(v);
            if(p == local) continue;
            part_t p_o = p % size;
            if(size < k){
                while(conn_entries(g_start + p_o) != NULL_PART && conn_entries(g_start + p_o) != p){
                    p_o = (p_o + 1) % size;
                }
            }
            if(conn_entries(g_start + p_o) == NULL_PART){
                conn_entries(g_start + p_o) = p;
                used_cap++;
            }
        }
        update += used_cap;
    }, result);
    return result;
}

static int64_t least_squares(const matrix_t g, part_vt part, const part_t k){
    gain_vt ex_cut = cut_per_part(g, part, k);
    int64_t result = 0;
    Kokkos::parallel_reduce("least squares", policy_t(0, k), KOKKOS_LAMBDA(const part_t p, int64_t& res){
        int64_t ex = ex_cut(p);
        res += ex*ex;
    }, result);
    return result;
}

static scalar_t sum(const wgt_view_t wdeg){
    scalar_t result = 0;
    Kokkos::parallel_reduce("sum view", policy_t(0, wdeg.size()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
        update += wdeg(i);
    }, result);
    // std::cout << "G degree: " << result << std::endl;
    return result;
}

};

}
