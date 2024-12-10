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
#include "defs.h"
#include "config.h"
#include "contract.hpp"
#include <sstream>
#include <string>
#include <iostream>

namespace jet_partitioner {

bool load_config(jet_partitioner::config_t& c, const char* config_f) {

    std::ifstream f(config_f);
    if (!f.is_open()) {
        std::cerr << "FATAL ERROR: Could not open config file " << config_f << std::endl;
        return false;
    }
    std::string lines[4];
    int reads = 0;
    // you might think that reading in four lines from a simple config file could be done like:
    // f >> c.coarsening_alg; 
    // f >> c.num_parts;
    // f >> c.num_iter;
    // f >> c.max_imb_ratio;
    // but that doesn't work if there are exactly 3 lines instead of 4 in the config file
    // because if the last line is a float like 3.14, then c.num_iter will contain the 3
    // and the c.max_imb_ratio will contain the .14
    for(int i = 0; i < 4; i++){
        if(f >> lines[i]) reads++;
    }
    f.close();
    if(reads != 4){
        std::cerr << "FATAL ERROR: Config file has less than 4 lines" << std::endl;
        return false;
    }
    c.coarsening_alg = std::stoi(lines[0]);
    c.num_parts = std::stoi(lines[1]);
    c.num_iter = std::stoi(lines[2]);
    c.max_imb_ratio = std::stod(lines[3]);
    return true;
}

template<typename t>
t fast_atoi( const char*& str )
{
    t val = 0;
    while(isdigit(*str)) {
        val = val*10 + static_cast<t>(*str - '0');
        str++;
    }
    return val;
}

void next_line(const char*& str){
    while(*str != '\n') str++;
    str++;
}

bool load_metis_graph(matrix_t& g, bool& uniform_ew, const char *fname) {
    Kokkos::Timer t;
    std::ifstream infp(fname, std::ios::binary);
    if (!infp.is_open()) {
        std::cerr << "FATAL ERROR: Could not open metis graph file " << fname << std::endl;
        return false;
    }
    infp.seekg(0, std::ios::end);
    size_t sz = infp.tellg();
    std::cout << "Reading " << sz << " bytes from " << fname << std::endl;
    infp.seekg(0, std::ios::beg);
    //1 for extra newline if needed
    char* s = new char[sz + 1];
    infp.read(s, sz);
    infp.close();
    std::cout << "Read graph from " << fname << " in " << std::setprecision(3) << t.seconds() << "s" << std::endl;
    //append an endline to end of file in case one doesn't exist
    //needed to prevent parser from overshooting end of buffer
    if(s[sz - 1] != '\n'){
        s[sz] = '\n';
        sz++;
    }
    const char* f = s;
    const char* fmax = s + sz;
    size_t header[4] = {0, 0, 0, 0};
    //ignore commented lines
    while(*f == '%') next_line(f);
    while(!isdigit(*f)) f++;
    //read header data
    for(int i = 0; i < 4; i++){
        header[i] = fast_atoi<size_t>(f);
        while(!isdigit(*f)){
            if(*f == '\n'){
                i = 4;
                f++;
                break;
            }
            f++;
        }
    }
    ordinal_t n = header[0];
    edge_offset_t m = header[1];
    int fmt = header[2];
    int ncon = header[3];
    bool has_ew = ((fmt % 10) == 1);
    if(fmt != 0 && fmt != 1){
        std::cerr << "FATAL ERROR: Unsupported format flags " << fmt << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        return false;
    }
    if(ncon != 0){
        std::cerr << "FATAL ERROR: Unsupported ncon " << ncon << std::endl;
        std::cerr << "Graph parser does not currently support vertex weights" << std::endl;
        return false;
    }
    vtx_view_t entries(Kokkos::ViewAllocateWithoutInitializing("entries"), m*2);
    vtx_mirror_t entries_m = Kokkos::create_mirror_view(entries);
    edge_view_t row_map(Kokkos::ViewAllocateWithoutInitializing("row map"), n + 1);
    edge_mirror_t row_map_m = Kokkos::create_mirror_view(row_map);
    wgt_view_t values(Kokkos::ViewAllocateWithoutInitializing("values"), 2*m);
    wgt_mirror_t values_m;
    if(has_ew){
        values_m = Kokkos::create_mirror_view(values);
    }
    edge_offset_t edges_read = 0;
    ordinal_t rows_read = 0;
    row_map_m(0) = 0;
    bool is_value = false;
    //read edge information
    while(f < fmax){
        //increment past whitespace
        while(f < fmax && !isdigit(*f)){
            //ignore commented lines
            if(*f == '%'){
                next_line(f);
                continue;
            }
            if(*f == '\n'){
                //ignore extra trailing newlines
                if(rows_read < n) row_map_m(++rows_read) = edges_read;
            }
            f++;
        }
        if(f >= fmax) break;
        //fast_atoi also increments past numeric chars
        ordinal_t edge_info = fast_atoi<ordinal_t>(f);
        if(!is_value){
            //subtract 1 to convert to 0-indexed
            entries_m(edges_read) = edge_info - 1;
        } else {
            values_m(edges_read++) = edge_info;
        }
        if(has_ew){
            is_value = !is_value;
        } else {
            edges_read++;
        }
    }
    delete[] s;
    if(rows_read != n || edges_read != 2*m){
        std::cerr << "FATAL ERROR: Mismatch between expected and actual line/nonzero count in metis file" << std::endl;
        std::cerr << "Read " << rows_read << " lines and " << edges_read << " nonzeros" << std::endl;
        std::cerr << "Lines expected: " << n << "; Nonzeros expected: " << m*2 << std::endl;
        return false;
    }
    Kokkos::deep_copy(row_map, row_map_m);
    Kokkos::deep_copy(entries, entries_m);
    if(has_ew){
        Kokkos::deep_copy(values, values_m);
        uniform_ew = false;
    } else {
        Kokkos::deep_copy(values, 1);
        uniform_ew = true;
    }
    graph_t g_graph(entries, row_map);
    g = matrix_t("input graph", n, values, g_graph);
    std::cout << "Processed graph at " << fname << " in " << std::setprecision(3) << t.seconds() << "s" << std::endl;
    return true;
}

void write_part(part_vt part_d, const char *fname){
    std::ofstream ofp(fname);
    if(!ofp.is_open()) return;
    part_mt part = Kokkos::create_mirror_view(part_d);
    Kokkos::deep_copy(part, part_d);
    size_t n = part.extent(0);
    std::stringstream ss;
    for(size_t x = 0; x < n; x++){
        ss << part(x) << std::endl;
    }
    ofp << ss.str();
    ofp.close();
}

part_vt load_part(ordinal_t n, const char *fname){
    std::ifstream ifp(fname);
    part_vt part_d("device part", n);
    if(!ifp.is_open()) return part_d;
    part_mt part = Kokkos::create_mirror_view(part_d);
    for(ordinal_t x = 0; x < n; x++){
        ifp >> part(x);
    }
    ifp.close();
    Kokkos::deep_copy(part_d, part);
    return part_d;
}

//loads a sequence of coarse graphs and mappings between each graph from binary file
//used to control for coarsening when experimenting with refinement
std::list<typename jet_partitioner::contracter<matrix_t>::coarse_level_triple> load_coarse(){
    using coarse_level_triple = typename jet_partitioner::contracter<matrix_t>::coarse_level_triple;
    std::list<coarse_level_triple> levels;
    std::list<coarse_level_triple> error_levels;
    FILE* cgfp = fopen("coarse_graphs.out", "r");
    int size = 0;
    if(fread(&size, sizeof(int), 1, cgfp) != 1) return error_levels;
    std::cout << "Importing " << size << " coarsened graphs" << std::endl;
    ordinal_t prev_n = 0;
    for(int i = 0; i < size; i++){
        coarse_level_triple level;
        level.level = i + 1;
        ordinal_t N = 0;
        if(fread(&N, sizeof(ordinal_t), 1, cgfp) != 1)  return error_levels;
        edge_offset_t M = 0;
        if(fread(&M, sizeof(edge_offset_t), 1, cgfp) != 1) return error_levels;
        edge_view_t rows("rows", N + 1);
        auto rows_m = Kokkos::create_mirror_view(rows);
        if(fread(rows_m.data(), sizeof(edge_offset_t), N + 1, cgfp) != static_cast<size_t>(N + 1)) return error_levels;
        Kokkos::deep_copy(rows, rows_m);
        vtx_view_t entries("entries", M);
        auto entries_m = Kokkos::create_mirror_view(entries);
        if(fread(entries_m.data(), sizeof(ordinal_t), M, cgfp) != static_cast<size_t>(M)) return error_levels;
        Kokkos::deep_copy(entries, entries_m);
        wgt_view_t values("values", M);
        auto values_m = Kokkos::create_mirror_view(values);
        if(fread(values_m.data(), sizeof(value_t), M, cgfp) != static_cast<size_t>(M))  return error_levels;
        Kokkos::deep_copy(values, values_m);
        graph_t graph(entries, rows);
        matrix_t g("g", N, values, graph);
        level.mtx = g;
        wgt_view_t vtx_wgts("vtx wgts", N);
        auto vtx_wgts_m = Kokkos::create_mirror_view(vtx_wgts);
        if(fread(vtx_wgts_m.data(), sizeof(value_t), N, cgfp) != static_cast<size_t>(N)) return error_levels;
        Kokkos::deep_copy(vtx_wgts, vtx_wgts_m);
        level.vtx_w = vtx_wgts;
        if(level.level > 1){
            vtx_view_t i_entries("entries", prev_n);
            auto i_entries_m = Kokkos::create_mirror_view(i_entries);
            if(fread(i_entries_m.data(), sizeof(ordinal_t), prev_n, cgfp) != static_cast<size_t>(prev_n)) return error_levels;
            Kokkos::deep_copy(i_entries, i_entries_m);
            typename jet_partitioner::contracter<matrix_t>::coarse_map i_g;
            i_g.coarse_vtx = N;
            i_g.map = i_entries;
            level.interp_mtx = i_g;
        }
        prev_n = N;
        levels.push_back(level);
    }
    fclose(cgfp);
    return levels;
}

//reads part from binary file
part_vt load_coarse_part(ordinal_t n){
    FILE* cgfp = fopen("coarse_part.out", "r");
    part_vt part("part", n);
    auto part_m = Kokkos::create_mirror_view(part);
    if(fread(part_m.data(), sizeof(part_t), n, cgfp) != static_cast<size_t>(n)) return part;
    Kokkos::deep_copy(part, part_m);
    fclose(cgfp);
    return part;
}

};