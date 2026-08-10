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
#include "contract.hpp"
// #include "contract_community.hpp"
// #include "jet_refiner_community.hpp"
#include "uncoarsen.hpp"
#include "initial_partition.hpp"
#include "memory_store.hpp"
#include "coarse_level.h"
#include "coarse_map.h"
#include "cluster_data.hpp"
#include "clustering_methods.h"
#include "weighted_graph.h"

namespace jet_partitioner {

template<class crsMat, typename part_t>
class partitioner {
public:

    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using exec_space = typename matrix_t::execution_space;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using coarsener_t = contracter<matrix_t>;
    // using comm_coarsener_t = jet_community::contracter<matrix_t>;
    using init_t = initial_partitioner<matrix_t, part_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_t = coarse_level<matrix_t>;
    using stat = part_stat<matrix_t, part_t>;
    using mem_t = memory_store<matrix_t, part_t>;
    using wg_t = jet_community::weighted_graph;

static wgt_vt degree_weighting(const matrix_t& g){
    wgt_vt vweights("weights", g.numRows());
    Kokkos::parallel_for("set v weights", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
    return vweights;
}

static part_vt partition(scalar_t& edge_cut,
                                  const config_t& config,
                                  const matrix_t g,
                                  const wgt_vt vweights,
                                  bool uniform_ew,
                                  experiment_data<scalar_t>& experiment,
                                  float lambda) {

    coarsener_t coarsener;
    switch(config.coarsening_alg){
        case 0:
            coarsener.set_heuristic(coarsener_t::MtMetis);
            break;
        case 1:
            coarsener.set_heuristic(coarsener_t::HECv1);
            break;
        case 2:
            coarsener.set_heuristic(coarsener_t::Match);
            break;
        default:
            coarsener.set_heuristic(coarsener_t::MtMetis);
    }
    part_t k = config.num_parts;
    ordinal_t opt = stat::optimal_size(g.numRows(), k);
    ordinal_t upper = opt*config.max_imb_ratio;
    ordinal_t cluster_limit = upper / 8;
    int cutoff = k*8;
    if(cutoff > 1024){
        cutoff = k*2;
        cutoff = std::max(1024, cutoff);
    }
    coarsener.set_coarse_vtx_cutoff(cutoff);
    coarsener.set_min_allowed_vtx(cutoff / 4);

    Kokkos::fence();
    Kokkos::Timer t;
    double start_time = t.seconds();
    double fin_uncoarsening = 0;
    part_vt part;
    {
        wg_t top;
        top.mtx = g;
        top.vtx_w = vweights;
        top.v_pen = degree_weighting(g);
        top.edge_uniform = true;
        top.reuse_w_as_pen = false;
        modularity rfd(top, lambda);
        mem_t mem(g, k, rfd);
        vtx_vt dummy_constraint;//("constraint", g.numRows());
        std::list<coarse_level_t> cg_list = jet_community::clustering_methods::leiden_part<false, false>(mem, top, rfd, dummy_constraint, cluster_limit, upper, cutoff * 2);
        // std::list<coarse_level_t> cg_list = coarsener.template generate_coarse_graphs<false>(top.mtx, top.vtx_w, mem, experiment, g.numRows(), dummy_constraint, true);
        // std::list<coarse_level_t> cg_list_part2 = coarsener.template generate_coarse_graphs<false>(cg_list.back().mtx, cg_list.back().vtx_w, mem, experiment, upper, dummy_constraint, false);
        // cg_list_part2.pop_front();
        // cg_list.splice(cg_list.end(), cg_list_part2);
        Kokkos::fence();
        double fin_coarsening_time = t.seconds();
        experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
        double imb_ratio = config.max_imb_ratio;
        part_vt coarsest_p = init_t::metis_init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
        // part_vt coarsest_p = init_t::init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
        // part_vt coarsest_p = init_t::random_init(cg_list.back().vtx_w, k, imb_ratio);
        Kokkos::fence();
        experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
        part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, config,
            edge_cut, mem, experiment);
        Kokkos::fence();
        fin_uncoarsening = t.seconds();
    }
    Kokkos::fence();
    // currently an error with huge-bubbles-0000 and large lambda
    for(int i = 0; i < 0; i++) {
        wg_t top;
        top.mtx = g;
        top.vtx_w = vweights;
        top.v_pen = vweights;
        top.edge_uniform = true;
        top.reuse_w_as_pen = true;
        normalized_lcc rfd(top, lambda);
        mem_t mem(g, k, rfd);
        vtx_vt constraint("constraint", g.numRows());
        Kokkos::deep_copy(constraint, part);
        std::list<coarse_level_t> cg_list = jet_community::clustering_methods::leiden_part<false, true>(mem, top, rfd, constraint, cluster_limit, upper, cutoff);
        Kokkos::fence();
        part = uncoarsener_t::uncoarsen(cg_list, constraint, config,
            edge_cut, mem, experiment);
        Kokkos::fence();
        fin_uncoarsening = t.seconds();
    }
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);
    experiment.addMeasurement(Measurement::FreeGraph, fin_time - fin_uncoarsening);
    
    if(config.verbose){
        // additional partition statistics
        // experiment.setMaxPartCut(stat::max_part_cut(g, part, k));
        // experiment.setObjective(stat::comm_size(g, part, k));

        // std::cout << "Verify cut: " << stat::get_total_cut(g, part) / 2 << std::endl;

        experiment.refinementReport();
        experiment.verboseReport();
    }

    return part;
}
};

}
