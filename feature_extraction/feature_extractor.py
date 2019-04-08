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
def calculate_results(path, filename, label=None):
    try:
        encoding = chardet.detect(open(path+filename, 'rb').read())
        if os.path.getsize(path+filename) > 0:
            with open(path+filename, 'r', encoding=encoding['encoding'] if encoding['encoding'] is not None else "utf-8") as f:
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
            rules = yara.compile('php.yar')
            matches = rules.match(path+filename)
            yara_match_counts = {}
            time.sleep(2)
            for x in matches: yara_match_counts["{}".format(x)]=len(x.strings)
            yara_results = create_yara_results(yara_match_counts)
            res =  [generate_filename(), file_size, ic, entropy, nasty_sig_count, super_nasty_count, uses_eval, compression_ratio, longest_word_length]
            res.extend(yara_results)
            res.append(label)
            return res 
    except Exception as e :
        print(traceback.format_exc())
        return None

def generate_filename():
    m = hashlib.sha1()
    m.update(str(random.randint(1,10000000)).encode('utf-8'))
    return m.hexdigest()[:10]	

def create_yara_results(yara_match_counts):
    yara_rules = ["NonPrintableChars", "ObfuscatedPhp", "PasswordProtection", "HiddenInAFile", "DangerousPhp", "DodgyPhp", "SuspiciousEncoding", "DodgyStrings", "CloudFlareBypass", "Websites"]
    results_array = []
    for rule in yara_rules:
        count = yara_match_counts.get(rule)
        if count is None:
            count = 0
        results_array.append(count)
    return results_array

if __name__ == "__main__":

    if sys.argv[1] == 'feature_extractor':
        with open('../datasets/backdoor_webshells_features_extended.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "File_Size", "Index_of_Coincidence", "Entropy", "Nasty_Sig_Count", "Super_Nasty_Sig_Count", "Eval_Uses", "Compression_Ratio", "Longest_Word_Length", "NonPrintableChars", "ObfuscatedPHP", "PasswordProtection", "HiddenInAFile", "DangerousPHP", "DodgyPHP", "SuspiciousEncoding", "DodgyStrings", "CloudflareBypass", "Websites", "Label"])
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

    if sys.argv[1] == 'feature_extractor_list':
        with open('../datasets/backdoor_webshells_features_extended_2.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "File_Size", "Index_of_Coincidence", "Entropy", "Nasty_Sig_Count", "Super_Nasty_Sig_Count", "Eval_Uses", "Compression_Ratio", "Longest_Word_Length", "NonPrintableChars", "ObfuscatedPHP", "PasswordProtection", "HiddenInAFile", "DangerousPHP", "DodgyPHP", "SuspiciousEncoding", "DodgyStrings", "CloudflareBypass", "Websites", "Label"])
            count = 0
            webshells = open("all_shells.txt",'r') 
            for f in webshells.readlines():
                f = f.strip('\n')
                print("Starting Extraction on: {}: {}".format(f, count))
                try: 
                    writer.writerow(calculate_results("", f, 1)) 
                    count += 1
                except Exception as e:
                    print ("Could not extract: {}".format(f))
            non_webshells = open("all_non_webshells.txt",'r') 
            for f in non_webshells.readlines():
                f = f.strip('\n')
                print("Starting Extraction on: {}: {}".format(f, count)) 
                try:
                    writer.writerow(calculate_results("", f, 0)) 
                    count += 1
                except Exception as e:
                    print("Could not extract: {}".format(f))

