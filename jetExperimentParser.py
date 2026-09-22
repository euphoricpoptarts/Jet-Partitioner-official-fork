import sys
import os
import re
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from itertools import zip_longest
import math
import csv

def getGraphList():
    data = []
    with open("graphlist.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    return data

stemParse = r"(.+)_(fm|spec|fullpar|compare)_(.*)_Sampling_Data"
lineParse = "{}mean={}, median={}, min={}, max={}, std-dev={}"
#fieldHeaders = "Graph, HEC, MIS, Match, MT"
fieldHeaders = "{Graph} & {hec} & {match} & {mtmetis} & {gosh} & {mis2}"
dnf = "{OOM}"

def getStats(filepath):
    with open(filepath,"r") as f:
       return json.load(f) 

def gpuBuildTableAlt(graphs,data,outFile):
    with open(outFile,"w+") as f:
        print("GPU Build Comparison Table", file=f)
        for graph in sorted(data, key=str.casefold):
            values = data[graph]
            l = [graph]
            exp = values[("fullpar","hec")]
            l.append("{:.0f}".format(exp["edge-cut-0"]["median"]))
            l.append("{:.3f}".format(exp["reftimes0"]["median"]))
            l.append("{:.3f}".format(exp["coarsen-duration-seconds"]["median"]))
            l.append("{:.3f}".format(exp["initial-partition-duration-seconds"]["median"]))
            #l.append("{:.3f}".format(exp["finest-refinement-duration-seconds1"]["median"]))
            #l.append("{:.3f}".format(exp["total-duration-seconds"]["median"]))
            print(" ".join(l), file=f)


def gpuBuildTable(graphs,data,expname,outFile):
    with open(outFile,"w+") as f:
        for graph, graphSanitized in getGraphList():
            l = [graphSanitized]
            if graph in data:
                values = data[graph]
                expkey = ("fullpar", expname)
                if expkey in values:
                    exp = values[expkey]
                    l.append("{:.0f}".format(exp["edge-cut"]["median"]))
                    l.append("{:.3f}".format(exp["edge-cut"]["std-dev"]))
                    # l.append("{:.0f}".format(exp["objective"]["median"]))
                    # l.append("{:.3f}".format(exp["coarse-edge-ratio"]["median"]))
                    #l.append("{:.0f}".format(exp["edge-cut-1"]["median"]))
                    #l.append("{:.0f}".format(exp["edge-cut-2"]["median"]))
                    #l.append("{:.3f}".format(exp["ratio1-v-fm"]["median"]))
                    #l.append("{:.3f}".format(exp["ratio2-v-fm"]["median"]))
                    rtime = exp["refine-duration-seconds"]["median"]
                    # rtimeF = exp["finest-refinement-duration-seconds"]["median"]
                    ctime = exp["coarsen-duration-seconds"]["median"]
                    itime = exp["initial-partition-duration-seconds"]["median"]
                    # l.append("{:.3f}".format(rtimeF))
                    l.append("{:.3f}".format(rtime))
                    l.append("{:.3f}".format(ctime))
                    l.append("{:.3f}".format(itime))
                    #l.append("{:.3f}".format(exp["reftimes1"]["median"]))
                    #l.append("{:.3f}".format(exp["reftimes2"]["median"]))
                    #l.append("{:.3f}".format(exp["coarsen-duration-seconds"]["median"]))
                    l.append("{:.3f}".format(rtime + ctime + itime))
                    l.append("{:.4f}".format(exp["total-duration-seconds"]["median"]))
                    l.append("{:.4f}".format(exp["number-coarse-levels"]["median"]))
                    #l.append("{:.3f}".format(exp["finest-refinement-duration-seconds1"]["median"]))
                    #l.append("{:.3f}".format(exp["total-duration-seconds"]["median"]))
                    print(" ".join(l), file=f)
                else:
                    print("{} nan nan nan nan nan nan nan nan".format(graphSanitized), file=f)
            else:
                print("{} nan nan nan nan nan nan nan nan".format(graphSanitized), file=f)

def main():

    logDir = sys.argv[1]
    outDir = sys.argv[2]

    globMatch = "{}/*.json".format(logDir)

    data = {}
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        stemMatch = re.match(stemParse, stem)
        if stemMatch is not None:
            graph = stemMatch.groups()[0]
            experiment = (stemMatch.groups()[1], stemMatch.groups()[2])
            if graph not in data:
                data[graph] = {}
            data[graph][experiment] = getStats(filepath)

    graphsSorted = [key for key in data]
    graphsSorted = sorted(graphsSorted, key = str.casefold)

    ks = [4, 16, 64]
    configs = ["i3louvain_lcalc",
               "i3leiden_calc",
               "i3match",
               "i3louvain_lp",
               "i3leiden_lp",
               "i3louvain_lp_stricter",
               "i3leiden_lp_stricter",
               "i3louvain_lp_strictest",
               "i3leiden_lp_strictest",
               "i3louvain_lp_mega_strictest",
               "i3leiden_lp_mega_strictest",
               "i3louvain_lp_stricter2",
               "i3leiden_lp_stricter2",
               "i3louvain_lp_strictest2",
               "i3leiden_lp_strictest2",
               "i3louvain_lp_mega_strictest2",
               "i3leiden_lp_mega_strictest2"]
    for p in range(1, 18):
        l = pow(2, p)
        config = "i3louvain_lp_limit_retry_{}".format(l)
        configs.append(config)
        config = "i3leiden_lp_limit_retry_{}".format(l)
        configs.append(config)
    for p in range(-3, 21):
        l = pow(2, p)
        config = "i3louvain_mod_l{}".format(l)
        configs.append(config)
        config = "i3leiden_mod_l{}".format(l)
        configs.append(config)
        config = "i3louvain_nlcc_l{}".format(l)
        configs.append(config)
        config = "i3leiden_nlcc_l{}".format(l)
        configs.append(config)
    for li in range(1, 10):
            l = li / 10.0
            config = "i3louvain_cpm_l{}".format(l)
            configs.append(config)
            config = "i3leiden_cpm_l{}".format(l)
            configs.append(config)
    for config in configs:
        for k in ks:
            gpuBuildTable(graphsSorted, data, f"hec_k{k}{config}", f"{outDir}/results_k{k}{config}.txt")

if __name__ == "__main__":
    main()
