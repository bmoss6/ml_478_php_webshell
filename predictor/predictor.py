from neopi import Entropy
from neopi import LanguageIC
from neopi import LongestWord
from neopi import SignatureNasty
from neopi import SignatureSuperNasty
from neopi import UsesEval
from neopi import Compression
import csv
import os
import chardet
import sys
import yara
import traceback
from joblib import load
import time
import random
import hashlib

    elif sys.argv[1] == "predict":
        clf = load("../saved_models/gradient_boosting.joblib")
        with open("neopi_ml.results", "w") as w:
            w.write("Filename\t Malicious Probability\n")
        for f in os.listdir(sys.argv[2]):
            if os.path.isfile(sys.argv[2]+f) and os.path.splitext(sys.argv[2]+f)[1]=='.php':
                results = calculate_results(sys.argv[2], f, None)
                if results is not None:
                    results = [results[1:-1]]
                    prob = clf.predict_proba(results)
                    print("{}:{}".format(f,prob[0][1]))
                    #with open("neopi_ml.results", "a") as w:
                    #    w.write("{}/{}\t{}".format(sys.argv[2], f, round(prob[1]*100,3)))

    elif sys.argv[1] == "predict_list":
        print("predict list!")
        clf = load(sys.argv[4])
        with open(sys.argv[3], "w") as w:
            w.write("Filename\tWebshell\tProbability\n")
        prefix = ""
        f = open(sys.argv[2],'r')
        if "eis" in sys.argv[2]:
            prefix = ""
        else:
            prefix = ""
        for fi in f.readlines():
            results = calculate_results(prefix, fi.strip('\n'), None)
            if results is not None:
                results = [results[1:-1]]
                pred = clf.predict(results)
                prob = clf.predict_proba(results)
                if pred[0] == 1:
                    pred = 'Yes'
                else:
                    pred = 'No'
                print("{}:{} ({})".format(fi.strip('\n'),pred, prob[0][1]))
                if prob[0][1] >float(sys.argv[5]):
                    with open(sys.argv[3], "a") as w:
                        w.write("{}\t{}\t{}\n".format(fi, pred, round((prob[0][1])*100,3)))
        f.close()
