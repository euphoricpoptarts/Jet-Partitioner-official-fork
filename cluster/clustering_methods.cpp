// Copyright 2026 Michael S. Gilbert II
// SPDX-License-Identifier: Apache-2.0
#include "header/local_mover.h"
#include "header/contract.h"
#include "header/leidenR.h"
#include "header/ordering.h"
#include "header/clustering_methods.h"

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

    coarse_level_t wg_to_level(const wg_t& wg){
        coarse_level_t level;
        level.mtx = wg.mtx;
        level.vtx_w = wg.vtx_w;
        level.uniform_weights = wg.edge_uniform;
        level.wdeg = wg.vtx_w;
        return level;
    }

    coarse_level_t wg_to_level(const wg_t& wg, vtx_vt coarse_map, ordinal_t coarse_vtx_count){
        coarse_level_t nx_out = wg_to_level(wg);
        typename coarse_level_t::coarse_map_t cm;
        cm.coarse_vtx = coarse_vtx_count;
        cm.map = coarse_map;
        nx_out.interp_mtx = cm;
        return nx_out;
    }

    void check_overwgt(const wgt_vt vtx_w, const scalar_t upper_bound){
        ordinal_t total_overwgt = 0;
        Kokkos::parallel_reduce("check overweight", policy_t(0, vtx_w.extent(0)), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update){
            if(vtx_w(i) > upper_bound) update++;
        }, total_overwgt);
        if(total_overwgt > 0) {
            std::cout << "Total overweight vertices: " << total_overwgt << "/" << vtx_w.extent(0) << std::endl;
        }
    }

    template <bool is_leiden>
    wg_t generate_next_level(const wg_t c, vtx_vt coarse_map, ordinal_t coarse_vtx_count, mem_t& mem, rfd_t& rfd){
        wg_t next_level;
        if(c.edge_uniform) next_level = contract_t::build_coarse_graph<true, !is_leiden>(c, coarse_map, coarse_vtx_count, mem);
        else next_level = contract_t::build_coarse_graph<false, false>(c, coarse_map, coarse_vtx_count, mem);
        // don't put this line before the graph contraction as that overwrites next_level
        next_level.reuse_w_as_pen = c.reuse_w_as_pen;
        next_level.vtx_w = wgt_vt("next level vtx weights", coarse_vtx_count);
        if(next_level.reuse_w_as_pen) next_level.v_pen = next_level.vtx_w;
        else next_level.v_pen = wgt_vt("next level vtx penalties", coarse_vtx_count);
        if(is_leiden) {
            coarsen_vtx_w(c.vtx_w, next_level.vtx_w, coarse_map);
            if(!next_level.reuse_w_as_pen) coarsen_vtx_w(c.v_pen, next_level.v_pen, coarse_map);
        } else {
            wgt_vt tw_rfd = Kokkos::subview(rfd.total_wgt, std::make_pair((ordinal_t)0, coarse_vtx_count));
            Kokkos::deep_copy(next_level.vtx_w, tw_rfd);
            if(!next_level.reuse_w_as_pen) {
                wgt_vt td_rfd = Kokkos::subview(rfd.total_deg, std::make_pair((ordinal_t)0, coarse_vtx_count));
                Kokkos::deep_copy(next_level.v_pen, td_rfd);
            }
        }
        return next_level;
    }

    template <bool plus, bool constrained>
    std::list<coarse_level_t> leiden_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& constraint, const ordinal_t upper_bound_in, const ordinal_t upper_bound_max, const ordinal_t target){
        std::vector<wg_t> levels;
        std::list<coarse_level_t> output;
        output.push_back(wg_to_level(top));
        levels.push_back(top);
        ordinal_t upper_bound = upper_bound_in;
        vtx_vt part("cluster assignments", top.mtx.numRows());
        Kokkos::parallel_for("set initial assignments", policy_t(0, top.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
            part(x) = x;
        });
        int limit = 0;
        int last_add = 0;
        bool imp = false;
        rfd_t copy(rfd);
        while(levels[levels.size() - 1].mtx.numRows() > target && limit++ < 100) {
            wg_t c = levels[levels.size() - 1];
            // orderings must be generated for use in local_move and build_coarse_graph
            order::generate_orderings(mem, c.mtx);
            copy.copy(rfd);
            vtx_vt pcopy(Kokkos::ViewAllocateWithoutInitializing("part copy"), c.mtx.numRows());
            Kokkos::deep_copy(pcopy, part);
            lm_t::local_move<constrained>(c, part, rfd, (levels.size() == 1), mem, constraint, true, upper_bound);
            vtx_vt louv = part;
            int coarse_vtx_count = 0;
            vtx_vt coarse_map;
            if(levels.size() == 1 && c.edge_uniform) coarse_map = lr_t::template coarsen_leidenR<true, true>(c, louv, mem, rfd, coarse_vtx_count);
            else if(levels.size() == 1 && !(c.edge_uniform)) coarse_map = lr_t::template coarsen_leidenR<true, false>(c, louv, mem, rfd, coarse_vtx_count);
            else coarse_map = lr_t::template coarsen_leidenR<false, false>(c, louv, mem, rfd, coarse_vtx_count);
            if(coarse_vtx_count < c.mtx.numRows() * 0.9){
                wg_t next_level = generate_next_level<true>(c, coarse_map, coarse_vtx_count, mem, rfd);
                levels.push_back(next_level);
                part = vtx_vt("cluster assignments coarse", coarse_vtx_count);
                downsample(louv, part, coarse_map);
                coarse_level_t nx_out = wg_to_level(next_level, coarse_map, coarse_vtx_count);
                output.push_back(nx_out);
                if(constrained) {
                    vtx_vt next_constraint("next constraint", coarse_vtx_count);
                    downsample(constraint, next_constraint, coarse_map);
                    constraint = next_constraint;
                }
                last_add = 0;
                imp = true;
            } else {
                if(imp){
                    upper_bound = upper_bound * 2;
                    if(upper_bound > upper_bound_max) upper_bound = upper_bound_max;
                }
                rfd.copy(copy);
                Kokkos::deep_copy(part, pcopy);
                last_add++;
                rfd.lambda /= 1.5;
                rfd.update_objective();
                // std::cout << "upper bound: " << upper_bound << std::endl;
                // std::cout << "lambda: " << rfd.lambda << std::endl;
                // std::cout << "vtx count: " << c.mtx.numRows() << std::endl;
                // setzero = false;
            }
        }
        std::cout << "Iterations since last added level: " << last_add << std::endl;
        std::cout << "Total iterations: " << limit << std::endl;

        return output;
    }

    template <bool constrained>
    std::list<coarse_level_t> louvain_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& constraint, const ordinal_t upper_bound_in, const ordinal_t upper_bound_max, const ordinal_t target){
        if(constrained) rfd.reset(top);
        std::vector<wg_t> levels;
        std::list<coarse_level_t> output;
        output.push_back(wg_to_level(top));
        levels.push_back(top);
        bool drop_constraint = false;//constrained;
        ordinal_t upper_bound = upper_bound_in;
        int limit = 0;
        bool grow_upper = false;
        rfd_t copy(rfd);
        while(levels[levels.size() - 1].mtx.numRows() > target && limit++ < 100) {
            wg_t c = levels[levels.size() - 1];
            vtx_vt part("cluster assignments", c.mtx.numRows());
            Kokkos::parallel_for("set initial assignments", policy_t(0, c.mtx.numRows()), KOKKOS_LAMBDA(const ordinal_t x){
                part(x) = x;
            });
            // orderings must be generated for use in local_move and build_coarse_graph
            order::generate_orderings(mem, c.mtx);
            copy.copy(rfd);
            lm_t::local_move<constrained>(c, part, rfd, true, mem, constraint, true, upper_bound);
            // the user clearly cares about quality if they are doing multiple iterations
            // if(constrained && rfd.label_count == c.mtx.numRows()){
            //     lm_t::local_move_strict<constrained>(c, part, rfd, true, mem, constraint);
            // }
            if(rfd.label_count < c.mtx.numRows()*0.9){
                wg_t next_level = generate_next_level<false>(c, part, rfd.label_count, mem, rfd);
                levels.push_back(next_level);

                coarse_level_t nx_out = wg_to_level(next_level, part, rfd.label_count);
                output.push_back(nx_out);

                if(constrained) {
                    vtx_vt next_constraint("next constraint", rfd.label_count);
                    downsample(constraint, next_constraint, part);
                    constraint = next_constraint;
                }
                grow_upper = true;
            } else {
                rfd.copy(copy);
                if(grow_upper) {
                    upper_bound = upper_bound * 2;
                    if(upper_bound > upper_bound_max) upper_bound = upper_bound_max;
                }
            }
            rfd.lambda /= 1.5;
            rfd.update_objective();
        }
        
        return output;
    }

    template <bool constrained>
    std::list<coarse_level_t> louvain_plus_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound){
        if(constrained) rfd.reset(top);
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
                wg_t next_level = generate_next_level<false>(c, part, rfd.label_count, mem, rfd);
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
        // int counter = 1;
        // for(auto& l : output){
        //     l.level = counter++;
        // }
        return output;
    }

    // explicit template instantiations
    template std::list<coarse_level_t> leiden_part<true, true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& input, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);
    template std::list<coarse_level_t> leiden_part<true, false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& input, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);
    template std::list<coarse_level_t> leiden_part<false, true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& input, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);
    template std::list<coarse_level_t> leiden_part<false, false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& input, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);

    template std::list<coarse_level_t> louvain_part<true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& constraint, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);
    template std::list<coarse_level_t> louvain_part<false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt& constraint, const ordinal_t upper_bound, const ordinal_t upper_bound_max, const ordinal_t target);
    template std::list<coarse_level_t> louvain_plus_part<true>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
    template std::list<coarse_level_t> louvain_plus_part<false>(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);
}

}