#pragma once
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_partitioner {

// this struct contains almost all auxiliary memory used by the algorithm
// this allows for efficient reuse of memory
template <class crsMat, typename part_t>
struct memory_store {

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using edge_vt = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using obj_vt = Kokkos::View<float*, Device>;
    static const ordinal_t max_sections = 128;
    static const int max_buckets = 50;

    // this struct contains memory which either requires initialization or some degree of persistence
    struct persistent {
        edge_vt row_map;
        gain_vt vals;
        vtx_vt entries;
        part_vt p_entries;
        part_vt sizes;
        vtx_vt cluster_sizes;
        gain_vt gain_persistent;
        obj_vt obj_persistent;
        gain_vt pvals;
        part_vt part, dest_part;
        vtx_vt lock_bit;
        vtx_vt order1, order2;
        part_vt dest_cache;
        ordinal_t offset_mid, offset_large;

        persistent(const matrix_t largest){
            ordinal_t n = largest.numRows();
            vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("vals"), largest.nnz());
            entries = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("entries"), largest.nnz());
            sizes = part_vt(Kokkos::ViewAllocateWithoutInitializing("table sizes"), n);
            if constexpr(std::is_same_v<ordinal_t, part_t>) {
                // reuse entries view if possible
                p_entries = entries;
                cluster_sizes = sizes;
            } else {
                p_entries = part_vt(Kokkos::ViewAllocateWithoutInitializing("part entries"), largest.nnz());
                cluster_sizes = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("cluster table sizes"), n);
            }
            row_map = edge_vt(Kokkos::ViewAllocateWithoutInitializing("row map"), n + 1);
            gain_persistent = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            obj_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            pvals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
            dest_part = part_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
            part = part_vt(Kokkos::ViewAllocateWithoutInitializing("part scratch"), n);
            lock_bit = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("lock bit"), n);
            dest_cache = part_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
            order1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 1"), n);
            order2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 2"), n);
        }
    };

    // this struct contains memory which can be used as-is
    struct scratch {
        gain_vt gain1, gain2, evict_start, evict_end, evict_fix, evict_diff;
        vtx_vt vtx1, vtx2, vtx3, zeros1;
        part_vt undersized;
        vtx_pin_st scan_host, pin_host;
        gain_pin_st cut_change1, cut_change2, max_part;
        gain_pin_vt reduce_locs;
        part_svt total_undersized;
        gain_svt max_vwgt;

        scratch(const ordinal_t n, const part_t k) {
            ordinal_t min_size = 1 + k*max_sections*max_buckets;
            gain1 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 1"), std::max(n, min_size));
            gain2 = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain scratch 2"), n);
            evict_start = gain_vt("evict start", k + 1);
            evict_end = gain_vt("evict end", k);
            evict_fix = gain_vt("evict fix", k);
            undersized = part_vt("undersized parts", k);
            vtx1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
            vtx2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), std::max(n, min_size));
            vtx3 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 3"), n);
            zeros1 = vtx_vt("zeros 1", n);
            scan_host = vtx_pin_st("scan host");
            pin_host = vtx_pin_st("pin host");
            total_undersized = part_svt("total undersized");
            max_vwgt = gain_svt("max vwgt allowed");
            reduce_locs = gain_pin_vt("reduce to here", 3);
            cut_change1 = Kokkos::subview(reduce_locs, 0);
            cut_change2 = Kokkos::subview(reduce_locs, 1);
            max_part = Kokkos::subview(reduce_locs, 2);
        }
    };

    persistent p_mem;
    scratch s_mem;

    memory_store(const matrix_t largest, part_t k) :
        p_mem(largest), 
        s_mem(largest.numRows(), k) {}
};

}