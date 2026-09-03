// Copyright 2026 Michael S. Gilbert II
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "cluster_data.hpp"
#include "core_types.h"

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
    using exec_space = typename matrix_t::execution_space;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using edge_vt = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using gain_svt = Kokkos::View<gain_t, Device>;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using edge_pin_st = Kokkos::View<edge_offset_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using part_svt = Kokkos::View<part_t, Device>;
    using obj_vt = Kokkos::View<float*, Device>;
    static const ordinal_t max_sections = 32;
    static const int max_buckets = 100;

    // this struct contains memory which either requires initialization or some degree of persistence
    struct persistent {
        edge_vt row_map;
        wgt_vt vals;
        vtx_vt entries;
        vtx_vt cluster_sizes;
        obj_vt obj_persistent;
        wgt_vt pvals, pvals_clone;
        vtx_vt part, dest_part;
        vtx_vt dest_cache;
        part_vt p_entries;
        part_vt sizes;
        gain_vt gain_persistent;
        vtx_vt lock_bit;

        persistent(const matrix_t largest){
            ordinal_t n = largest.numRows();
            vals = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("vals"), largest.nnz()*1.2);
            entries = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("entries"), largest.nnz()*1.2);
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
            obj_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            pvals = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
            pvals_clone = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("p vals clone"), n);
            dest_part = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
            part = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("part scratch"), n);
            gain_persistent = gain_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            lock_bit = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("lock bit"), n);
            dest_cache = part_vt(Kokkos::ViewAllocateWithoutInitializing("best connected part for each vertex"), n);
        }
    };

    struct ordering {
        vtx_vt order1, order2;
        ordinal_t last_scan_mid, last_scan_mid2, last_scan_large, last_scan_large2;
        ordinal_t offset_mid, offset_mid2, offset_large, offset_large2;

        ordering(const ordinal_t n){
            order1 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 1"), n);
            order2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 2"), n);
        }
    };

    // this struct contains memory which can be used as-is
    struct scratch {
        vtx_vt vtx1, vtx2, vtx3, zeros1;
        vtx_pin_st scan_host, pin_host, pin_host2;
        edge_pin_st edge_scan_host;
        gain_vt gain1, gain2, evict_start, evict_end, evict_fix, evict_diff;
        part_vt undersized;
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
            vtx2 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), n);
            vtx3 = vtx_vt(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 3"), n);
            zeros1 = vtx_vt("zeros 1", n);
            scan_host = vtx_pin_st("scan host");
            edge_scan_host = edge_pin_st("edge scan host");
            pin_host = vtx_pin_st("pin host");
            pin_host2 = vtx_pin_st("pin host 2");
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
    ordering o_mem;
    cluster_data spare_cluster_data;
    exec_space s1, s2;
    cudaEvent_t e0, e1, e2;

    void wait_default(){
        cudaEventRecord(e0, exec_space().cuda_stream());
        cudaStreamWaitEvent(s1.cuda_stream(), e0, 0);
        cudaStreamWaitEvent(s2.cuda_stream(), e0, 0);
    }

    void align_streams(){
        cudaStream_t s0 = exec_space().cuda_stream();
        cudaEventRecord(e1, s1.cuda_stream());
        cudaEventRecord(e2, s2.cuda_stream());
        cudaStreamWaitEvent(s0, e1, 0);
        cudaStreamWaitEvent(s0, e2, 0);
    }

    memory_store(const matrix_t largest, part_t k, cluster_data& clone_target) :
        p_mem(largest),
        s_mem(largest.numRows(), k),
        o_mem(largest.numRows()),
        spare_cluster_data(clone_target) {
            auto [s1c, s2c] = Kokkos::Experimental::partition_space(exec_space(),1,1);
            s1 = s1c;
            s2 = s2c;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventCreate(&e2);
        }

    memory_store(const memory_store&) = delete;

    ~memory_store(){
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
    }
};