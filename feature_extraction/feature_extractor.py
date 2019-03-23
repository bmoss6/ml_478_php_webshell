from feature_extraction.neopi import Entropy
from feature_extraction.neopi import LanguageIC
from feature_extraction.neopi import LongestWord
from feature_extraction.neopi import SignatureNasty
from feature_extraction.neopi import SignatureSuperNasty
from feature_extraction.neopi import UsesEval
from feature_extraction.neopi import Compression
import csv
import os
import chardet
import sys
from joblib import load

def calculate_results(path, filename, label=None):
    encoding = chardet.detect(open(path+filename, 'rb').read())
    with open(path+filename, 'r', encoding=encoding['encoding']) as f:
        ic = LanguageIC().calculate(f.read(), filename)
        f.seek(0)
        entropy = Entropy().calculate(f.read(), filename)
        f.seek(0)
        nasty_sig_count = SignatureNasty().calculate(f.read(), filename)
        f.seek(0)
        super_nasty_count = SignatureSuperNasty().calculate(f.read(), filename)
        f.seek(0)
        uses_eval = UsesEval().calculate(f.read(), filename)
        f.seek(0)
        longest_word_length = LongestWord().calculate(f.read(), filename)
    with open(path+filename, 'rb') as f:
        compression_ratio = Compression().calculate(f.read(), filename)
        file_size = os.path.getsize(path+filename)
        return [filename, file_size, ic, entropy, nasty_sig_count, super_nasty_count, uses_eval, compression_ratio, longest_word_length, label]



if __name__ == "__main__":

    if sys.argv[1] == 'feature_extractor':
        with open('../datasets/backdoor_webshells_features.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "File_Size", "Index_of_Coincidence", "Entropy", "Nasty_Sig_Count", "Super_Nasty_Sig_Count", "Eval_Uses", "Compression_Ratio", "Longest_Word_Length", "Label"])
            count = 0
            for f in os.listdir("../data_sources/php_webshells"):
                print("Starting Extraction on: {}: {}".format(f, count))
                try: 
                    writer.writerow(calculate_results("../data_sources/php_webshells/", f, 1)) 
                    count += 1
                except Exception as e:
                    print ("Could not extract: {}".format(f))
            for f in os.listdir("../data_sources/non_webshells"):
                print("Starting Extraction on: {}: {}".format(f, count)) 
                try:
                    writer.writerow(calculate_results("../data_sources/non_webshells/", f, 0)) 
                    count += 1
                except Exception as e:
                    print("Could not extract: {}".format(f))

    elif sys.argv[1] == "predict":
        clf = load("../saved_models/gradient_boosting.joblib")
        with open("neopi_ml.results", "w") as w:
            w.write("Filename\t Malicious Probability\n")
        for f in os.lisdir(sys.argv[2]):
            results = calculate_results(sys.argv[2], f, None)
            results = [results[1:-1]]
            prob = clf.predict_proba(results)
            print(prob[1])
            with open("neopi_ml.results", "a") as w:
                w.write("{}/{}\t{}".format(sys.argv[2], f, round(prob[1]*100,3)))
