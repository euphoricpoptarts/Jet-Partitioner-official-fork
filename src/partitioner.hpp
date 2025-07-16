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
#include "contract_community.hpp"
#include "jet_refiner_community.hpp"
#include "memory_store_community.hpp"
#include "uncoarsen.hpp"
#include "initial_partition.hpp"
#include "memory_store.hpp"

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
    using comm_coarsener_t = jet_community::contracter<matrix_t>;
    using init_t = initial_partitioner<matrix_t, part_t>;
    using uncoarsener_t = uncoarsener<matrix_t, part_t>;
    using coarse_level_triple = typename coarsener_t::coarse_level_triple;
    using clt = typename comm_coarsener_t::coarse_level_triple;
    using stat = part_stat<matrix_t, part_t>;
    using mem_t = memory_store<matrix_t, part_t>;
    using comm_mem_t = jet_community::memory_store<matrix_t, ordinal_t>;

static wgt_vt degree_weighting(const matrix_t& g){
    wgt_vt vweights("weights", g.numRows());
    Kokkos::parallel_for("set v weights", policy_t(0, g.numRows()), KOKKOS_LAMBDA(const ordinal_t i){
        vweights(i) = g.graph.row_map(i + 1) - g.graph.row_map(i);
    });
    return vweights;
}

static void coarsen_vtx_w(wgt_vt in, wgt_vt out, vtx_vt map){
    Kokkos::parallel_for("set v weights", policy_t(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
        ordinal_t c = map(i);
        Kokkos::atomic_add(&out(c), in(i));
    });
}

static std::list<coarse_level_triple> louvain_part(matrix_t g, ordinal_t upper, ordinal_t cutoff){
    using ref_t = jet_community::jet_refiner<matrix_t, ordinal_t>;
    using rfd_t = typename ref_t::refine_data;
    using coarse_map = typename coarsener_t::coarse_map;
    comm_mem_t mem(g);
    clt top;
    top.mtx = g;
    top.wdeg = degree_weighting(g);
    std::vector<clt> levels;
    std::vector<coarse_map> parts;
    wgt_vt input_vtx_w("input vertex weights", g.numRows());
    Kokkos::deep_copy(input_vtx_w, 1);
    levels.push_back(top);
    rfd_t rfd;
    rfd.init = false;
    ref_t refiner;
    while(true) {
        clt c = levels[levels.size() - 1];
        vtx_vt part("cluster assignments", c.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", r_policy(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        if(levels.size() == 1) refiner.template jet_refine<true>(c.mtx, c.wdeg, input_vtx_w, part, rfd, true, upper, mem);
        else refiner.template jet_refine<false>(c.mtx, c.wdeg, input_vtx_w, part, rfd, true, upper, mem);
        coarse_map cm;
        cm.map = part;
        cm.coarse_vtx = rfd.label_count;
        parts.push_back(cm);
        if(rfd.label_count < c.mtx.numRows() && rfd.label_count >= cutoff){
            comm_coarsener_t contracter;
            wgt_vt next_input_vtx_w("next input vertex weights", rfd.label_count);
            coarsen_vtx_w(input_vtx_w, next_input_vtx_w, part);
            input_vtx_w = next_input_vtx_w;
            clt next_clt = contracter.build_coarse_graph(c, part, rfd.label_count, mem);
            next_clt.wdeg = wgt_vt("weighted degree 2", rfd.label_count);
            Kokkos::deep_copy(next_clt.wdeg, rfd.total_deg);
            levels.push_back(next_clt);
        } else {
            break;
        }
    }

    std::vector<coarse_level_triple> coarse;
    for(size_t i = 0; i < levels.size(); i++){
        coarse_level_triple level;
        clt knockoff = levels[i];
        level.mtx = knockoff.mtx;
        if(i > 0) level.interp_mtx = parts[i - 1];
        level.level = i+1;
        level.uniform_weights = (i == 0);
        level.vtx_w = wgt_vt("vtx weights", level.mtx.numRows());
        if(i == 0){
            Kokkos::deep_copy(level.vtx_w, 1);
        } else {
            coarsen_vtx_w(coarse[i - 1].vtx_w, level.vtx_w, level.interp_mtx.map);
        }
        coarse.push_back(level);
    }

    std::list cl(coarse.begin(), coarse.end());
    return cl;
}

static part_vt partition(scalar_t& edge_cut,
                                  const config_t& config,
                                  const matrix_t g,
                                  const wgt_vt vweights,
                                  bool uniform_ew,
                                  experiment_data<scalar_t>& experiment) {

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
        mem_t mem(g, k);
        std::list<coarse_level_triple> cg_list;
        if(true) cg_list = louvain_part(g, upper / 4, cutoff);
        else coarsener.generate_coarse_graphs(g, vweights, mem, experiment, uniform_ew);
        Kokkos::fence();
        double fin_coarsening_time = t.seconds();
        experiment.addMeasurement(Measurement::Coarsen, fin_coarsening_time - start_time);
        double imb_ratio = config.max_imb_ratio;
        part_vt coarsest_p = init_t::metis_init(cg_list.back().mtx, cg_list.back().vtx_w, k, imb_ratio);
        //part_vt coarsest_p = init_t::random_init(cg_list.back().vtx_w, k, imb_ratio);
        Kokkos::fence();
        experiment.addMeasurement(Measurement::InitPartition, t.seconds() - fin_coarsening_time);
        part = uncoarsener_t::uncoarsen(cg_list, coarsest_p, config,
            edge_cut, mem, experiment);
        Kokkos::fence();
        fin_uncoarsening = t.seconds();
    }
    Kokkos::fence();
    double fin_time = t.seconds();
    experiment.addMeasurement(Measurement::Total, fin_time - start_time);
    experiment.addMeasurement(Measurement::FreeGraph, fin_time - fin_uncoarsening);
    
    if(config.verbose){
        // additional partition statistics
        experiment.setMaxPartCut(stat::max_part_cut(g, part, k));
        experiment.setObjective(stat::comm_size(g, part, k));

        experiment.refinementReport();
        experiment.verboseReport();
    }

    return part;
}
};

}
