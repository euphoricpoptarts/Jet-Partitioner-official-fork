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
#include <limits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "coarse_map.h"
#include "cluster_data.hpp"

namespace jet_community {

template<typename ordinal_t>
KOKKOS_INLINE_FUNCTION ordinal_t xorshiftHash(ordinal_t key) {
  ordinal_t x = key;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

template<class crsMat>
class matching {
public:
    // define internal types
    using matrix_t = crsMat;
    using exec_space = typename matrix_t::execution_space;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    using vtx_vt = typename Kokkos::View<ordinal_t*, Device>;
    using wgt_vt = typename Kokkos::View<scalar_t*, Device>;
    using policy_t = typename Kokkos::RangePolicy<exec_space>;
    using team_policy_t = typename Kokkos::TeamPolicy<exec_space>;
    using member = typename team_policy_t::member_type;
    using pool_t = Kokkos::Random_XorShift64_Pool<Device>;
    using gen_t = typename pool_t::generator_type;
    using hasher_t = Kokkos::pod_hash<ordinal_t>;
    // there is a problem edge-case in kokkos with MaxLoc that can be triggered rarely for any input graph
    // the problem will be fixed soon, use MaxFirstLoc in meantime
    using argmax_reducer_t = Kokkos::MaxFirstLoc<uint32_t, edge_offset_t, Device>;
    using argmax_t = typename argmax_reducer_t::value_type;
    using coarse_map_t = coarse_map<vtx_vt>;
    using rfd_t = cluster_data;
    static constexpr ordinal_t ORD_MAX = std::numeric_limits<ordinal_t>::max();
    static constexpr bool is_host_space = std::is_same<typename exec_space::memory_space, typename Kokkos::DefaultHostExecutionSpace::memory_space>::value;    

    static ordinal_t countUnmatched(vtx_vt target) {
        ordinal_t total = 0;

        Kokkos::parallel_reduce("count unmatched", policy_t(0, target.extent(0)), KOKKOS_LAMBDA(ordinal_t i, ordinal_t& update) {
            if (target(i) == ORD_MAX) {
                update++;
            }
        }, total);

        return total;
    }

    template<typename hash_t>
    static void matchHash(const vtx_vt unmappedVtx, const Kokkos::View<hash_t*, Device> hashes, const hash_t nullkey, vtx_vt vcmap, wgt_vt vtx_w, ordinal_t upper){
        ordinal_t mappable = unmappedVtx.extent(0);
        Kokkos::View<hash_t*, Device> htable(Kokkos::ViewAllocateWithoutInitializing("hashes hash table"), mappable);
        vtx_vt twins(Kokkos::ViewAllocateWithoutInitializing("twin table"), mappable);
        Kokkos::deep_copy(htable, nullkey);
        Kokkos::deep_copy(twins, -1);
        Kokkos::parallel_for("match by hash", policy_t(0, mappable), KOKKOS_LAMBDA(const ordinal_t x){
            ordinal_t i = unmappedVtx(x);
            hash_t h = hashes(x);
            if(h == nullkey) return;
            ordinal_t key = h % mappable;
            bool found = false;
            //find the slot already owned by key
            //or claim ownership of a slot for this key
            while(!found){
                if(htable(key) == nullkey){
                    Kokkos::atomic_compare_exchange(&htable(key), nullkey, h);
                }
                if(htable(key) == h){
                    found = true;
                } else {
                    key++;
                    if(key >= mappable) key -= mappable;
                }
            }
            found = false;
            //check if another vertex with same digest is in slot
            //if so, match with it
            //else, insert into slot
            while(!found){
                ordinal_t twin = twins(key);
                if(twin == -1){
                    if(Kokkos::atomic_compare_exchange(&twins(key), twin, i) == twin) found = true;
                } else {
                    if(Kokkos::atomic_compare_exchange(&twins(key), twin, -1) == twin){
                        ordinal_t cv = twin < i ? twin : i;
                        if(vtx_w(twin) + vtx_w(i) <= upper){
                            vcmap(twin) = cv;
                            vcmap(i) = cv;
                        }
                        found = true;
                    }
                }
            }
        });
    }

    template<bool is_initial, bool is_uniform>
    struct pickMatch {
        matrix_t g;
        vtx_vt vcmap;
        vtx_vt hn;
        pool_t rand_pool;
        vtx_vt vperm;
        ordinal_t n;
        ordinal_t perm_length;
        wgt_vt vtx_w;
        ordinal_t upper;
        vtx_vt constraint;

        pickMatch(matrix_t _g,
            vtx_vt _vcmap,
            vtx_vt _hn,
            pool_t _rand_pool,
            vtx_vt _vperm,
            ordinal_t _n,
            ordinal_t _perm_length,
            wgt_vt _vtx_w,
            ordinal_t _upper,
            vtx_vt _constraint) :
                g(_g),
                vcmap(_vcmap),
                hn(_hn),
                rand_pool(_rand_pool),
                vperm(_vperm),
                n(_n),
                perm_length(_perm_length),
                vtx_w(_vtx_w),
                upper(_upper),
                constraint(_constraint) {}

        KOKKOS_INLINE_FUNCTION
        void operator()(const member& thread) const {
            const ordinal_t i = thread.league_rank();
            ordinal_t u = perm_length == n ? i : vperm(i);
            if(!is_initial && (vcmap(u) != ORD_MAX || hn(u) == ORD_MAX || vcmap(hn(u)) == ORD_MAX)) return;
            scalar_t max_ewt = 0;
            edge_offset_t start = g.graph.row_map(u);
            edge_offset_t end = g.graph.row_map(u+1);
            uint32_t r = 0;
            Kokkos::single(Kokkos::PerTeam(thread), [=](uint32_t& update){
                gen_t generator = rand_pool.get_state();
                update = generator.urand();
                rand_pool.free_state(generator);
            }, r);
            if(!is_uniform){
                // find max edge weight
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, scalar_t& update){
                    ordinal_t v = g.graph.entries(j);
                    if(!is_initial && vcmap(v) != ORD_MAX) return;
                    if(g.values(j) > update && vtx_w(u) + vtx_w(v) <= upper && constraint(u) == constraint(v)){
                        update = g.values(j);
                    }
                }, Kokkos::Max<scalar_t, Device>(max_ewt));
            }
            thread.team_barrier();
            argmax_t argmax{0, end};
            // select a random adjacent vertex having the max edge weight
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(thread, start, end), [=](const edge_offset_t j, argmax_t& local) {
                //v must be unmatched to be considered
                uint32_t v = g.graph.entries(j);
                if(!is_initial && vcmap(v) != ORD_MAX) return;
                if(constraint(u) != constraint(v)) return;
                if(is_uniform || (g.values(j) == max_ewt  && vtx_w(u) + vtx_w(v) <= upper)){
                    uint32_t tiebreaker = xorshiftHash<uint32_t>(v + r);
                    // >= since 0 must be a valid max val
                    if(tiebreaker >= local.val){
                        local.val = tiebreaker;
                        local.loc = j;
                    }
                }
            }, argmax_reducer_t(argmax));
            thread.team_barrier();
            if(argmax.loc >= start && argmax.loc < end){
                ordinal_t hn_u = g.graph.entries(argmax.loc);
                hn(u) = hn_u;
            } else {
                hn(u) = ORD_MAX;
            }
        }

        KOKKOS_INLINE_FUNCTION
        void operator()(const ordinal_t& i) const {
            ordinal_t u = perm_length == n ? i : vperm(i);
            if(!is_initial && (vcmap(u) != ORD_MAX || hn(u) == ORD_MAX || vcmap(hn(u)) == ORD_MAX)) return;
            ordinal_t h = ORD_MAX;
            gen_t generator = rand_pool.get_state();
            uint32_t r = generator.urand();
            rand_pool.free_state(generator);
            scalar_t max_ewt = 0;
            uint32_t tiebreaker = 0;
            // select a random adjacent vertex having the max edge weight
            for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                ordinal_t v = g.graph.entries(j);
                if(constraint(u) != constraint(v)) continue;
                //v must be unmatched to be considered
                if (is_initial || vcmap(v) == ORD_MAX) {
                    if (!is_uniform && (max_ewt < g.values(j) && vtx_w(u) + vtx_w(v) <= upper)) {
                        max_ewt = g.values(j);
                        h = v;
                        tiebreaker = xorshiftHash<uint32_t>(v + r);
                    } else if(is_uniform || (max_ewt == g.values(j) && vtx_w(u) + vtx_w(v) <= upper)){
                        uint32_t sim_wgt = xorshiftHash<uint32_t>(v + r);
                        // >= since 0 must be a valid max tiebreaker
                        if(sim_wgt >= tiebreaker){
                            h = v;
                            tiebreaker = sim_wgt;
                        }
                    }
                }
            }
            hn(u) = h;
        }
    };

    static coarse_map_t coarsen_match(const matrix_t& g,
        const bool uniform_weights, pool_t& rand_pool,
        wgt_vt vtx_w,
        const ordinal_t upper,
        const vtx_vt constraint) {

        ordinal_t n = g.numRows();

        vtx_vt hn(Kokkos::ViewAllocateWithoutInitializing("heavies"), n);
        vtx_vt vcmap(Kokkos::ViewAllocateWithoutInitializing("vcmap"), n);
        Kokkos::deep_copy(hn, ORD_MAX);
        Kokkos::deep_copy(vcmap, ORD_MAX);
        vtx_vt vperm_scratch(Kokkos::ViewAllocateWithoutInitializing("vperm"), n);
        vtx_vt vperm = vperm_scratch;

        if (uniform_weights) {
            pickMatch<false, true> matcher(g, vcmap, hn, rand_pool, vperm, n, n, vtx_w, upper, constraint);
                if(!is_host_space && g.nnz() / g.numRows() > 32){
                    Kokkos::parallel_for("Potential matches (random)", team_policy_t(n, Kokkos::AUTO), matcher);
                } else {
                    Kokkos::parallel_for("Potential matches (random)", policy_t(0, n), matcher);
                }
        }
        else {
            pickMatch<true, false> matcher(g, vcmap, hn, rand_pool, vperm, n, n, vtx_w, upper, constraint);
            if(!is_host_space && g.nnz() / g.numRows() > 32){
                Kokkos::parallel_for("Potential matches (heavy)", team_policy_t(n, Kokkos::AUTO), matcher);
            } else {
                Kokkos::parallel_for("Potential matches (heavy)", policy_t(0, n), matcher);
            }
        }
        ordinal_t perm_length = n;
        //construct mapping using heaviest edges
        vtx_vt perm_scratch(Kokkos::ViewAllocateWithoutInitializing("next perm"), n);
        while (perm_length > 0) {
            //std::cout << "Remaining vtx: " << perm_length << std::endl;
            //match vertices with vertex given by hn
            for(int r = 0; r < 4; r++){
                // use hash to determine which vertices are active/inactive in each phase
                // vary the hash with r so that the active/inactive sets change greatly between phases
                // we want to match as many vertices as possible according to hn to avoid computing new matches
                // this approach seems to produce higher qualtiy partitions than a maximal independent set induced by hn
                Kokkos::parallel_for("commit matches (part 1)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                    ordinal_t u = perm_length == n ? i : vperm(i);
                    ordinal_t v = hn(u);
                    if(v == ORD_MAX || vcmap(u) != ORD_MAX) return;
                    hasher_t hash;
                    bool condition = false;
                    if(r > 0) condition = (hash(u + r) < hash(v + r));
                    else condition = (u < v);
                    // vertices passing condition are active
                    // vertices failing condition are inactive
                    // a match can only occur by an active vertex matching an inactive vertex
                    if (!condition) {
                        vcmap(u) = ORD_MAX - 1;
                    }
                });
                Kokkos::parallel_for("commit matches (part 2)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                    ordinal_t u = perm_length == n ? i : vperm(i);
                    ordinal_t v = hn(u);
                    if(v == ORD_MAX || vcmap(u) != ORD_MAX) return;
                    ordinal_t cv = u < v ? u : v;
                    if (Kokkos::atomic_compare_exchange(&vcmap(v), ORD_MAX - 1, cv) == ORD_MAX - 1) {
                        vcmap(u) = cv;
                    }
                });
                Kokkos::parallel_for("commit matches (part 3)", policy_t(0, perm_length), KOKKOS_LAMBDA(ordinal_t i){
                    ordinal_t u = perm_length == n ? i : vperm(i);
                    if(vcmap(u) == ORD_MAX - 1){
                        vcmap(u) = ORD_MAX;
                    }
                });
            }

            // find new matches for unmatched vertices
            if(uniform_weights){
                pickMatch<false, true> matcher(g, vcmap, hn, rand_pool, vperm, n, perm_length, vtx_w, upper, constraint);
                if(!is_host_space && g.nnz() / g.numRows() > 32){
                    Kokkos::parallel_for("Potential matches (random)", team_policy_t(perm_length, Kokkos::AUTO), matcher);
                } else {
                    Kokkos::parallel_for("Potential matches (random)", policy_t(0, perm_length), matcher);
                }
            } else {
                pickMatch<false, false> matcher(g, vcmap, hn, rand_pool, vperm, n, perm_length, vtx_w, upper, constraint);
                if(!is_host_space && g.nnz() / g.numRows() > 32){
                    Kokkos::parallel_for("Potential matches (heavy)", team_policy_t(perm_length, Kokkos::AUTO), matcher);
                } else {
                    Kokkos::parallel_for("Potential matches (heavy)", policy_t(0, perm_length), matcher);
                }
            }
            vtx_vt perm = perm_scratch;
            if(perm_length != n){
                perm = Kokkos::subview(perm_scratch, std::make_pair((ordinal_t)0, perm_length));
                Kokkos::deep_copy(exec_space(), perm, vperm);
            }
            Kokkos::parallel_scan("scan remaining", policy_t(0, perm_length), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                ordinal_t u = perm_length == n ? i : perm(i);
                if(vcmap(u) == ORD_MAX && hn(u) != ORD_MAX){
                    if(final){
                        vperm_scratch(update) = u;
                    }
                    update++;
                }
            }, perm_length);
            vperm = Kokkos::subview(vperm_scratch, std::make_pair((ordinal_t)0, perm_length));
        }

        if (true) {
            ordinal_t unmapped = countUnmatched(vcmap);
            double unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //leaf matches
            if (unmappedRatio > 0.25) {
                vtx_vt unmappedVtx(Kokkos::ViewAllocateWithoutInitializing("unmapped vertices"), unmapped);
                ordinal_t mappable;
                Kokkos::parallel_scan("scan unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                    if(vcmap(i) == ORD_MAX && g.graph.row_map(i+1) - g.graph.row_map(i) == 1){
                        ordinal_t v = g.graph.entries(g.graph.row_map(i));
                        if(constraint(i) == constraint(v)){
                            if(final){
                                unmappedVtx(update) = i;
                            }
                            update++;
                        }
                    }
                }, mappable);
                unmappedVtx = Kokkos::subview(unmappedVtx, std::make_pair((ordinal_t)0, mappable));
                vtx_vt hashes(Kokkos::ViewAllocateWithoutInitializing("hashes"), mappable);
                Kokkos::parallel_for("create digests", policy_t(0, mappable), KOKKOS_LAMBDA(ordinal_t i) {
                    ordinal_t u = unmappedVtx(i);
                    ordinal_t v = g.graph.entries(g.graph.row_map(u));
                    hashes(i) = v;
                });
                ordinal_t nullkey = ORD_MAX;
                matchHash<ordinal_t>(unmappedVtx, hashes, nullkey, vcmap, vtx_w, upper);
            }

            unmapped = countUnmatched(vcmap);
            unmappedRatio = static_cast<double>(unmapped) / static_cast<double>(n);

            //relative matches
            if (unmappedRatio > 0.25) {
                vtx_vt unmappedVtx(Kokkos::ViewAllocateWithoutInitializing("unmapped vertices"), unmapped);
                ordinal_t mappable;
                Kokkos::parallel_scan("scan unmapped", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
                    if(vcmap(i) == ORD_MAX){
                        if(final){
                            unmappedVtx(update) = i;
                        }
                        update++;
                    }
                }, mappable);
                vtx_vt hashes(Kokkos::ViewAllocateWithoutInitializing("hashes"), mappable);
                Kokkos::parallel_for("create digests", policy_t(0, mappable), KOKKOS_LAMBDA(ordinal_t i) {
                    ordinal_t u = unmappedVtx(i);
                    ordinal_t h = ORD_MAX;
                    scalar_t max_wgt = 0;
                    ordinal_t min_deg = ORD_MAX;

                    // select the lowest degree adjacent vertex
                    for (edge_offset_t j = g.graph.row_map(u); j < g.graph.row_map(u + 1); j++) {
                        ordinal_t v = g.graph.entries(j);
                        if(constraint(u) != constraint(v)) continue;
                        ordinal_t vdeg = g.graph.row_map(v+1) - g.graph.row_map(v);
                        if (min_deg > vdeg) {
                            min_deg = vdeg;
                            max_wgt = g.values(j);
                            h = v;
                        } else if(min_deg == vdeg){
                            if(max_wgt < g.values(j)){
                                h = v;
                                max_wgt = g.values(j);
                            }
                        }
                    }
                    hashes(i) = h;
                });
                ordinal_t nullkey = ORD_MAX;
                matchHash<ordinal_t>(unmappedVtx, hashes, nullkey, vcmap, vtx_w, upper);
            }
        }

        //create singleton aggregates of remaining unmatched vertices
        Kokkos::parallel_for(policy_t(0, n), KOKKOS_LAMBDA(ordinal_t i){
            if (vcmap(i) == ORD_MAX) {
                vcmap(i) = i;
            }
        });
        ordinal_t nc = 0;
        //if something breaks here it's probably cuz adding n causes overflow
        Kokkos::parallel_scan("set coarse ids", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i, ordinal_t& update, const bool final){
            if(vcmap(i) == i){
                if(final){
                    vcmap(i) = update;
                }
                update++;
            } else if(final) {
                vcmap(i) += n;
            }
        }, nc);
        Kokkos::parallel_for("prop coarse ids", policy_t(0, n), KOKKOS_LAMBDA(const ordinal_t i){
            if(vcmap(i) >= n){
                vcmap(i) = vcmap(vcmap(i) - n);
            }
        });

        coarse_map_t out;
        out.coarse_vtx = nc;
        out.map = vcmap;
        return out;
    }
};

}
