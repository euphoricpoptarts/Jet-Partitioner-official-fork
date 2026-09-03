import sys
import os
import subprocess
from glob import glob
from parse import parse
from statistics import mean, stdev, median
import secrets
import json
from pathlib import Path
from threading import Thread, BoundedSemaphore
from itertools import zip_longest
import csv

def getGraphList():
    data = []
    with open("graphlist.csv", "r") as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)
    return data

callTuple = ("./build/app/jet","fullpar_hec")

rateLimit = BoundedSemaphore(value = 1)
waitLimit = 3600

def printStat(fieldTitle, statList, outfile):
    min_s = min(statList)
    max_s = max(statList)
    avg = mean(statList)
    sdev = "only one data-point"
    if len(statList) > 1:
        sdev = stdev(statList)
    med = median(statList)
    print("{}: mean={}, median={}, min={}, max={}, std-dev={}".format(fieldTitle, avg, med, min_s, max_s, sdev), file=outfile)

def listToStats(statList):
    stats = {}
    stats["min"] = min(statList)
    stats["max"] = max(statList)
    stats["mean"] = mean(statList)
    stats["std-dev"] = "only one data-point"
    if len(statList) > 1:
        stats["std-dev"] = stdev(statList)
    stats["median"] = median(statList)
    return stats

def dictToStats(data):
    output = {}
    for key, value in data.items():
        if len(value) > 0 and isinstance(value[0], dict):
            d = []
            for datum in value:
                d.append(dictToStats(datum))
            output[key] = d
        elif len(value) > 0:
            output[key] = listToStats(value)
    return output


def printDict(data, outfile):
    for key, value in data.items():
        if len(value) > 0 and isinstance(value[0], dict):
            for idx, datum in enumerate(value):
                print("{} Level {}:".format(key, idx), file=outfile)
                printDict(datum, outfile)
        elif len(value) > 0:
            printStat(key, value, outfile)

def transposeListOfDicts(data):
    data = [x for x in data if x is not None]
    transposed = {}
    if len(data) == 0:
        return transposed
    #all entries should have same fields
    fields = [x for x in data[0]]
    for field in fields:
        transposed[field] = [datum[field] for datum in data]

    fieldsToTranpose = []
    for key, value in transposed.items():
        if len(value) > 0 and isinstance(value[0], list):
            fieldsToTranpose.append(key)

    for field in fieldsToTranpose:
        #value is a list of lists, transform it into list of dicts
        aligned_lists = zip_longest(*transposed[field])
        dict_list = []
        for l in aligned_lists:
            d = transposeListOfDicts(l)
            dict_list.append(d)
        transposed[field] = dict_list
    return transposed

def analyzeMetrics(metricsPath, logFile):
    with open(metricsPath, "r") as fp:
        data = json.load(fp)

    data = transposeListOfDicts(data)

    with open(logFile, "w") as output:
        printDict(data, output)

    statsDict = dictToStats(data)
    jsonFile = os.path.splitext(logFile)[0] + ".json"
    with open(jsonFile, "w") as output:
        json.dump(statsDict, output)

def runExperiment(executable, filepath, metricDir, logFile, config, l):

    if(os.path.exists(logFile)):
        return

    exe_string = parse("./{}", executable)[0]
    if(not os.path.exists(exe_string)):
        print("Error: Could not find executable {}".format(exe_string))
        return
    giveup = False
    while giveup is not True:
        metricsPath = "{}/group{}.txt".format(metricDir, secrets.token_urlsafe(10))
        call = [executable, filepath, config, l, "/dev/null", metricsPath]
        call_str = " ".join(call)
        with rateLimit:
            print("running {}".format(call_str), flush=True)
            stdout_f = "tmp_log.txt"
            with open(stdout_f, 'w') as fp:
                process = subprocess.Popen(call, stdout=fp, stderr=subprocess.DEVNULL)
            try:
                returncode = process.wait(timeout = waitLimit)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Timeout reached by {}".format(call_str), flush=True)
                return

        if(returncode != 0):
            if returncode != -11:
                giveup = True
            print("error code: {}".format(returncode))
            print("error produced by:")
            print(call_str, flush=True)
        else:
            analyzeMetrics(metricsPath, logFile)
            return

def make_config(fname, k, alg):
    with open(fname, "w") as f:
        print(alg, file=f)
        print(str(k), file=f)
        print("35", file=f)
        print("1.03", file=f)
        print("0", file=f)

def processGraph(filepath, metricDir, logFilePrefix):

    ks = [4, 16, 64]
    configs = [("tmp_config.txt", "i3louvain_lcalc", "7", "1"),
               ("tmp_config.txt", "i3leiden_calc", "11", "1"),
               ("tmp_config.txt", "i3match", "0", "1")]
    for p in range(-3, 21):
        l = pow(2, p)
        config = ("tmp_config.txt", "i3louvain_mod_l{}".format(l), "4", str(l))
        configs.append(config)
        config = ("tmp_config.txt", "i3leiden_mod_l{}".format(l), "8", str(l))
        configs.append(config)
        config = ("tmp_config.txt", "i3louvain_nlcc_l{}".format(l), "5", str(l))
        configs.append(config)
        config = ("tmp_config.txt", "i3leiden_nlcc_l{}".format(l), "9", str(l))
        configs.append(config)
    for li in range(1, 10):
            l = li / 10.0
            config = ("tmp_config.txt", "i3louvain_cpm_l{}".format(l), "6", str(l))
            configs.append(config)
            config = ("tmp_config.txt", "i3leiden_cpm_l{}".format(l), "10", str(l))
            configs.append(config)
    call, name = callTuple
    for config, cname, alg, l in configs:
        for k in ks:
            cname = f"k{k}{cname}"
            make_config(config, k, alg)
            logFile = "{}_{}_{}_Sampling_Data.txt".format(logFilePrefix, name, cname)
            print(cname)
            runExperiment(call, filepath, metricDir, logFile, config, l)

    print("end {} processing".format(filepath), flush=True)

def reprocessMetricsFromLogFile(f_path):
    form = "running {} sgpar on csr/{}.csr, data logged in {}"
    reprocessList = []
    with open(f_path) as fp:
        for line in fp:
            r = parse(form, line)
            if r != None:
                reprocess = {}
                reprocess['metrics'] = r[2]
                reprocess['log'] = "redo_stats/" + r[1] + "_" + r[0].replace(" ","_") + ".txt"
                reprocessList.append(reprocess)

    for reprocess in reprocessList:
        print(reprocess)
        try:
            analyzeMetrics(reprocess['metrics'], reprocess['log'])
        except:
            print("Couldn't process last")

def main():

    dirpath = sys.argv[1]
    metricDir = sys.argv[2]
    logDir = sys.argv[3]
    globMatch = "{}/*.graph".format(dirpath)

    matches = {}
    for file in glob(globMatch):
        filepath = file
        stem = Path(filepath).stem
        matches[stem] = filepath

    for stem, oname in getGraphList():
        #will fill in the third argument later
        logFile = "{}/{}".format(logDir, stem)
        processGraph(matches[stem], metricDir, logFile)

if __name__ == "__main__":
    main()
