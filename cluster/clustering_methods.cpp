// Copyright 2026 Michael S. Gilbert II
// SPDX-License-Identifier: Apache-2.0
#include "header/local_mover.h"
#include "header/contract.h"
#include "header/leidenR.h"
#include "header/ordering.h"
#include "header/clustering_methods.h"
#include "match.hpp"

namespace jet_community {

namespace contract_t = contracter;
namespace lr_t = leidenR;
namespace lm_t = local_move_heuristic;
namespace order = ordering;

namespace clustering_methods {
    // define internal types
    using exec_space = typename matrix_t::execution_space;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = typename Kokkos::View<scalar_t*, Device>;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;

    void coarsen_vtx_w(wgt_vt in, wgt_vt out, vtx_vt map){
        Kokkos::parallel_for("set v weights", policy_t(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t c = map(i);
            Kokkos::atomic_add(&out(c), in(i));
        });
    }

    void downsample(vtx_vt in, vtx_vt out, vtx_vt map){
        Kokkos::parallel_for("set v weights", policy_t(0, in.extent(0)), KOKKOS_LAMBDA(const ordinal_t i){
            ordinal_t c = map(i);
            out(c) = in(i);
        });
    }

    coarse_level_t wg_to_level(wg_t& wg){
        coarse_level_t level;
        level.mtx = wg.mtx;
        level.vtx_w = wg.vtx_w;
        level.uniform_weights = wg.edge_uniform;
        level.wdeg = wg.vtx_w;
        return level;
    }

    template <bool plus, bool improve>
    std::list<coarse_level_t> leiden_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound_in){
        std::vector<wg_t> levels;
        std::list<coarse_level_t> output;
        output.push_back(wg_to_level(top));
        levels.push_back(top);
        ordinal_t upper_bound = upper_bound_in;
        vtx_vt part("cluster assignments", top.mtx.numRows());
        if(!improve){
            Kokkos::parallel_for("set initial assignments", policy_t(0, top.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
        } else Kokkos::deep_copy(part, input);
        bool setzero = true;
        typename matching<matrix_t>::pool_t rand_pool(std::time(nullptr));
        while(true) {
            wg_t c = levels[levels.size() - 1];
            // double old_obj = rfd.obj;
            // orderings must be generated for use in local_move and build_coarse_graph
            order::generate_orderings(mem, c.mtx);
            lm_t::local_move<false>(c, part, rfd, !improve && (levels.size() == 1), mem, part, true, upper_bound);
            // if(rfd.label_count == c.mtx.numRows()){
            //     break;
            // }
            vtx_vt louv = part;
            typename coarse_level_t::coarse_map_t cm = matching<matrix_t>::coarsen_match(c.mtx, c.edge_uniform, rand_pool, c.vtx_w, upper_bound, louv);
            int coarse_vtx_count = cm.coarse_vtx;
            // if(levels.size() == 1 && c.edge_uniform) coarse_map = lr_t::template coarsen_leidenR<true, true>(c, louv, mem, rfd, coarse_vtx_count);
            // else if(levels.size() == 1 && !(c.edge_uniform)) coarse_map = lr_t::template coarsen_leidenR<true, false>(c, louv, mem, rfd, coarse_vtx_count);
            // else coarse_map = lr_t::template coarsen_leidenR<false, false>(c, louv, mem, rfd, coarse_vtx_count);
            if(coarse_vtx_count < c.mtx.numRows() * 0.9){
                wg_t next_level;
                if(c.edge_uniform) next_level = contract_t::build_coarse_graph<true, false>(c, cm.map, coarse_vtx_count, mem);
                else next_level = contract_t::build_coarse_graph<false, false>(c, cm.map, coarse_vtx_count, mem);
                next_level.vtx_w = wgt_vt("weighted degree 2", coarse_vtx_count);
                coarsen_vtx_w(c.vtx_w, next_level.vtx_w, cm.map);

                part = vtx_vt("cluster assignments coarse", coarse_vtx_count);
                downsample(louv, part, cm.map);
                levels.push_back(next_level);
                coarse_level_t nx_out = wg_to_level(next_level);
                // typename coarse_level_t::coarse_map_t cm;
                // cm.coarse_vtx = coarse_vtx_count;
                // cm.map = coarse_map;
                nx_out.interp_mtx = cm;
                output.push_back(nx_out);
            } else if (setzero) {
                rfd.lambda = 0;
                rfd.update_objective();
                upper_bound = upper_bound_in * 2;
                setzero = false;
            } else {
                break;
            }
        }

        return output;
    }

    template <bool constrained>
    std::list<coarse_level_t> louvain_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound){
        if(constrained) rfd.reset(top.mtx, top.vtx_w);
        std::vector<wg_t> levels;
        std::list<coarse_level_t> output;
        output.push_back(wg_to_level(top));
        levels.push_back(top);
        bool drop_constraint = constrained;
        while(true) {
            wg_t c = levels[levels.size() - 1];
            vtx_vt part("cluster assignments", c.mtx.numRows());
            Kokkos::parallel_for("set initial assignments", policy_t(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
            // orderings must be generated for use in local_move and build_coarse_graph
            order::generate_orderings(mem, c.mtx);
            lm_t::local_move<constrained>(c, part, rfd, true, mem, constraint, true, upper_bound);
            // the user clearly cares about quality if they are doing multiple iterations
            if(constrained && rfd.label_count == c.mtx.numRows()){
                lm_t::local_move_strict<constrained>(c, part, rfd, true, mem, constraint);
            }
            if(rfd.label_count < c.mtx.numRows()*0.9){
                wg_t next_level;
                if(c.edge_uniform) next_level = contract_t::build_coarse_graph<true, true>(c, part, rfd.label_count, mem);
                else next_level = contract_t::build_coarse_graph<false, false>(c, part, rfd.label_count, mem);
                wgt_vt td_rfd = Kokkos::subview(rfd.total_deg, std::make_pair((ordinal_t)0, rfd.label_count));
                next_level.vtx_w = wgt_vt("next level vtx weights", rfd.label_count);
                Kokkos::deep_copy(next_level.vtx_w, td_rfd);
                levels.push_back(next_level);
                coarse_level_t nx_out = wg_to_level(next_level);
                typename coarse_level_t::coarse_map_t cm;
                cm.coarse_vtx = rfd.label_count;
                cm.map = part;
                nx_out.interp_mtx = cm;
                output.push_back(nx_out);

                if(constrained) {
                    vtx_vt next_constraint("next constraint", rfd.label_count);
                    downsample(constraint, next_constraint, part);
                    constraint = next_constraint;
                }
            } else if(drop_constraint) {
                Kokkos::deep_copy(constraint, 0);
                drop_constraint = false;
            } else {
                break;
            }
        }

        int counter = 1;
        for(auto& l : output){
            l.level = counter++;
        }
        
        return output;
    }

    template <bool constrained>
    std::list<coarse_level_t> louvain_plus_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound){
        if(constrained) rfd.reset(top.mtx, top.vtx_w);
        std::vector<wg_t> levels;
        std::vector<vtx_vt> parts;
        levels.push_back(top);
        bool drop_constraint = constrained;
        while(true) {
            wg_t c = levels[levels.size() - 1];
            vtx_vt part("cluster assignments", c.mtx.numRows());
            Kokkos::parallel_for("set initial assignments", policy_t(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
            // orderings must be generated for use in local_move and build_coarse_graph
            order::generate_orderings(mem, c.mtx);
            lm_t::local_move<constrained>(c, part, rfd, true, mem, constraint, true, upper_bound);
            // the user clearly cares about quality if they are doing multiple iterations
            // if(constrained && rfd.label_count == c.mtx.numRows()){
            //     lm_t::local_move_strict<constrained>(c, part, rfd, true, mem, constraint);
            // }
            // this logic (when part is appended to vector) should be reworked
            parts.push_back(part);
            if(rfd.label_count < c.mtx.numRows()){
                wg_t next_level;
                if(c.edge_uniform) next_level = contract_t::build_coarse_graph<true, true>(c, part, rfd.label_count, mem);
                else next_level = contract_t::build_coarse_graph<false, false>(c, part, rfd.label_count, mem);
                wgt_vt td_rfd = Kokkos::subview(rfd.total_deg, std::make_pair((ordinal_t)0, rfd.label_count));
                next_level.vtx_w = wgt_vt("next level vtx weights", rfd.label_count);
                Kokkos::deep_copy(next_level.vtx_w, td_rfd);
                levels.push_back(next_level);

                if(constrained) {
                    vtx_vt next_constraint("next constraint", rfd.label_count);
                    downsample(constraint, next_constraint, part);
                    constraint = next_constraint;
                }
            } else if(drop_constraint) {
                Kokkos::deep_copy(constraint, 0);
                // this logic should be reworked
                parts.pop_back();
                drop_constraint = false;
            } else {
                break;
            }
        }

        if(levels.size() > 1){
            // last level has the same partition as previous level
            // so refining this level on the uncoarsening pass
            // would not integrate any coarse information
            levels.pop_back();
            parts.pop_back();
        }
        
        // levels.size()-2 so that (i+1) is in bounds
        for(int i = levels.size() - 2; i >= 0; i--){
            wg_t c = levels[i];
            vtx_vt coarse_part = parts[i + 1];
            vtx_vt part = parts[i];
            Kokkos::parallel_for("update top level assignments", policy_t(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = coarse_part(part(x));
            });
            order::generate_orderings(mem, c.mtx);
            lm_t::local_move<constrained>(c, part, rfd, false, mem, constraint, true, upper_bound);
        }

        std::list<coarse_level_t> output;
        output.push_back(wg_to_level(top));
        {
            wg_t coarsest;
            order::generate_orderings(mem, top.mtx);
            if(top.edge_uniform) coarsest = contract_t::build_coarse_graph<true, true>(top, parts[0], rfd.label_count, mem);
            else coarsest = contract_t::build_coarse_graph<false, false>(top, parts[0], rfd.label_count, mem);
            wgt_vt td_rfd = Kokkos::subview(rfd.total_deg, std::make_pair((ordinal_t)0, rfd.label_count));
            coarsest.vtx_w = wgt_vt("next level vtx weights", rfd.label_count);
            Kokkos::deep_copy(coarsest.vtx_w, td_rfd);
            coarse_level_t nx_out = wg_to_level(coarsest);
            typename coarse_level_t::coarse_map_t cm;
            cm.coarse_vtx = rfd.label_count;
            cm.map = parts[0];
            nx_out.interp_mtx = cm;
            output.push_back(nx_out);
        }
        int counter = 1;
        for(auto& l : output){
            l.level = counter++;
        }
        return output;
    }

    // explicit template instantiations
    template std::list<coarse_level_t> leiden_part<true, true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound);
    template std::list<coarse_level_t> leiden_part<true, false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound);
    template std::list<coarse_level_t> leiden_part<false, true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound);
    template std::list<coarse_level_t> leiden_part<false, false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound);

    template std::list<coarse_level_t> louvain_part<true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
    template std::list<coarse_level_t> louvain_part<false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
    template std::list<coarse_level_t> louvain_plus_part<true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
    template std::list<coarse_level_t> louvain_plus_part<false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
}

}