import numpy as np
from sklearn.utils import murmurhash3_32
import pandas as pd
from collections import Counter, defaultdict
import hashlib
import time
import os
import sys
import math
import argparse
import csv
import gc
sys.path.insert(0, "/home/sadiya/treeminhash/treeminhash/c++")
import tmh_pybind


import struct, hashlib

def tmh_pairs_to_tokens(sig, mode="ids"):

    ids, vals = [], []
    for item in sig:
        try:                       # tuple/list like (id, value)
            cid, cval = item[0], item[1]
        except (TypeError, KeyError):
            cid, cval = item.first, item.second   # pybind pair object
        ids.append(int(cid))
        vals.append(cval)

    if mode == "ids":
        return ids                                     # list[int]
    # id_value: MUST match the string the C++ writer put in the .txt files
    return [f"{i} {v}" for i, v in zip(ids, vals)]     # list[str] 

def load_tmh_txt(path,mode="raws"):
    # """TreeMinHash sketch stored as text: one slot per line -> '<id> <value>'."""
    #     """
    # Convert tmh_pybind.build_signature_from_cms output -> the SAME token form
    # that load_tmh_txt produces, so query and corpus are comparable.
    #   sig: list of (id, value) pairs (or objects with .first/.second)
    # """
    ids, raws = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, sval = line.split()
            ids.append(int(sid))
            raws.append(f"{sid} {sval}")    
              # exact canonical pair string
    if mode == "ids":
        return ids                            # id-only
    # id_value: hash the exact {id value} text so it matches iff BOTH match
    return raws
    
class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        self.hash_functions = [lambda x, seed=i: murmurhash3_32(str(x), seed=seed) % width for i in range(depth)]

    def add(self, key, count=1):
        for i, h in enumerate(self.hash_functions):
            self.table[i][h(key)] += count

    def query(self, key):
        return min(self.table[i][h(key)] for i, h in enumerate(self.hash_functions))

    
def cms_jaccard_similarity(cms_a, cms_b, depth, width):
    numerator = 0
    denominator = 0
    for i in range(depth):
        for j in range(width):
            numerator += min(cms_a.table[i][j], cms_b.table[i][j])
            denominator += max(cms_a.table[i][j], cms_b.table[i][j])
    return numerator / denominator if denominator > 0 else 0.0

def cms_sampling_jaccard_similarity(cms_a, cms_b, depth, width, sampling_ratio):
    numerator = 0
    denominator = 0
    for i in range(depth):
        for j in range(int(sampling_ratio*width)):
            numerator += min(cms_a.table[i][j], cms_b.table[i][j])
            denominator += max(cms_a.table[i][j], cms_b.table[i][j])
    return numerator / denominator if denominator > 0 else 0.0

def cms_earlystopping_jaccard_similarity(cms_a, cms_b, depth, width, threshold1):
        union = 0
        intersection = 0
        for i in range(depth):
            for j in range(int(width * 0.05)):
                union += max(cms_a.table[i][j], cms_b.table[i][j])
                intersection += min(cms_a.table[i][j], cms_b.table[i][j])
        initial_jaccard = intersection / union if union > 0 else 0.0
        if initial_jaccard < threshold1:
            return initial_jaccard
        union = 0
        intersection = 0
        for i in range(depth):
            for j in range(width):
                union += max(cms_a.table[i][j], cms_b.table[i][j])
                intersection += min(cms_a.table[i][j], cms_b.table[i][j])
        
        return intersection / union if union > 0 else 0.0

def actual_jaccard_similarity(set_a, set_b):
    set_a = [str(value) for value in set_a if pd.notna(value) and value != ""]
    set_b = [str(value) for value in set_b if pd.notna(value) and value != ""]
    freq_a = Counter(set_a)
    freq_b = Counter(set_b)
    
    intersection_keys = set(freq_a.keys()).intersection(set(freq_b.keys()))
    union_keys = set(freq_a.keys()).union(set(freq_b.keys()))
    
    numerator = sum(min(freq_a[key], freq_b[key]) for key in intersection_keys)
    denominator = sum(max(freq_a.get(key, 0), freq_b.get(key, 0)) for key in union_keys)
    
    return numerator / denominator if denominator > 0 else 0.0

def minhash_signature_weighted(cms, num_hashes, width, depth):
    signature = []
    for j in range(depth):
        for i in range(num_hashes):
            hash_vals = []
            for elem in range(1, width + 1):
                weight = cms.table[j][elem - 1]
                for k in range(weight):
                    hash_vals.append(murmurhash3_32(f"{elem}_{k}", seed=i))
            
            min_val = min(hash_vals) if hash_vals else 0  # Avoid empty case
            signature.append(min_val)
    return signature

def cms_minhash_jaccard_similarity(sig1, sig2):
    matches = 0
    for j in range(len(sig1)):
        if(sig1[j]==sig2[j]):
            matches+=1
    return matches/len(sig1)

def reduce_signature_size(input_folder, output_folder, sz):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if not filename.endswith('.txt'):
            continue
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        reduced_signature = []
        for line in lines[:]:
            nums = list(map(int, line.strip().split()))
            reduced_signature.append(nums[:sz])

        with open(output_path, 'w') as f:
            for row in reduced_signature:
                f.write(' '.join(map(str, row)) + '\n')

def hash_band(band):
    return hashlib.md5(str(band).encode()).hexdigest()

def build_lsh_index(signatures, num_bands):
    lsh_index = defaultdict(set)
    for id, signature in signatures.items():
        # print(id)
        # print(num_bands,len( signature))
        assert len(signature) % num_bands == 0, "Signature length must be divisible by num_bands"
        num_rows = len(signature) // num_bands
        for band_idx in range(num_bands):
            band = tuple(signature[band_idx*num_rows : (band_idx + 1)*num_rows])
            bucket_key = hash_band(band)
            lsh_index[bucket_key].add(id)
    return lsh_index

def find_similar_signatures(query_signature, num_bands, lsh_index):
    similar_columns = set()
    num_rows = len(query_signature) // num_bands

    for band_idx in range(num_bands):
        band = tuple(query_signature[band_idx * num_rows : (band_idx + 1) * num_rows])
        # print(band)
        band_hash = hash_band(band)
        if(band_hash in lsh_index):
            similar_columns.update(lsh_index[band_hash])
    return similar_columns

def find_optimal_bands(n, user_threshold, delta2):
    def get_divisors(n):
        divisors = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
        return sorted(divisors, reverse=True)
    divisors = get_divisors(n)
    optimal_number_bands = n
    for b in divisors:
        r = n // b
        if 1 + (math.log(delta2) // math.log(1-math.pow(user_threshold,r))) <= b:
            optimal_number_bands = b
        else:
            break
    return optimal_number_bands

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./nyc_cleaned",
                        help="Path to the dataset directory (no trailing /)")
    parser.add_argument("--dataset_name", type=str, default="nyc",
                        help="Name of the dataset")
    parser.add_argument("--width", type=int, default="nyc",
                        help="width")
    parser.add_argument("--depth", type=int, default="chembl.txt",
                        help="depth")
    parser.add_argument("--threshold", type=float, default="chembl.txt",
                        help="depth")
    parser.add_argument("--query_file", type=str, required=True, help="Name of the query file")
    parser.add_argument("--query_column", type=str, required=True, help="Name of the query column")
    parser.add_argument("--selectivity", type=str, default=0.1, help="Selectivity of the query")
    parser.add_argument("--run", type=int, default=10, help="run_number")
    args = parser.parse_args()
    
    query_file = args.query_file
    query_column = args.query_column
    query_selectivity = args.selectivity
    query_name = query_file.split('.')[0]
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    

    width = args.width
    depth = args.depth
    user_error = 0.063
    user_probability_minhash = 0.1
    user_probability_lsh = 0.05
    no_min_hash = (math.ceil((math.log(2/user_probability_minhash))/(2*user_error*user_error))+depth-1)//depth
    total_min_hash = no_min_hash * depth
    input_folder_minhash = f"/data/sadiya/tmh_cms_sketches/nyc_2000_5"
    # output_folder = f"/data/sadiya/nyc_2/minhash_signatures_{total_min_hash}_{dataset_name}"
    # reduce_signature_size(input_folder, output_folder, no_min_hash)
    # print("signatures trimmed")

    query_data = pd.read_csv(f'{dataset_path}/{query_file}', header=0, low_memory=False)[query_column].values
    query_cms = CountMinSketch(width, depth)
    for value in query_data:
        if pd.isna(value) or value == "": continue
        else:
            query_cms.add(value)
    query_minhash_creation_time =time.time()

    # NOTE: this builds one TreeMinHash signature over the WHOLE CMS matrix
    # (all depth rows + width cols flattened into a single weighted multiset),
    # not the per-row murmurhash scheme minhash_signature_weighted used.
    # Signature values here are NOT directly comparable to signature_b loaded
    # from input_folder/output_folder until the corpus side is regenerated
    # the same way. total_min_hash keeps the output length what the rest of
    # the pipeline (banding, len checks) currently expects.
    raw_sig = tmh_pybind.build_signature_from_cms(query_cms.table, total_min_hash)
    query_minhash_creation_time=time.time()- query_minhash_creation_time
    query_signature = tmh_pairs_to_tokens(raw_sig, "ids")   # match the corpus mode
    
    input_folder_cms = f"/data/sadiya/cms_sketch_chembl_2000_5/nyc_2000_5"
    # input_folder_minhash = output_fold
    # output_actual_doc_id_dir = f'{dataset_name}_output_doc_id/actual_doc_ids'
    # os.makedirs(output_actual_doc_id_dir, exist_ok=True)
    output_actual_doc_id_file = os.path.join("/home/sadiya/datadiscovery/nyc_output_doc_id/actual_doc_ids", f'{query_name}_{query_column}.txt')

    jaccard_threshold = args.threshold
    earlystop_threshold = 0.1
    sampling_ratio = 0.5

    all_docs_id = []
    actual_doc_id = []
    cms_doc_id = []
    cms_sampling_doc_id = []
    cms_earlystopping_doc_id = []
    cms_minhash_doc_id = []
    cms_minhash_lsh_doc_id = []

    with open(output_actual_doc_id_file, 'r') as f:
        actual_doc_id = [line.strip() for line in f if line.strip()]

    cms_time = 0
    cms_sampling_time = 0
    cms_earlystopping_time = 0
    cms_minhash_time = 0
    cms_minhash_lsh_time = 0
    failed_files=f"failed_w{width}_d{args.run}.txt"
    minhash_signatures = {}
    file_id = 1
    for file in sorted(os.listdir(dataset_path)):
        # if(file_id<2099):
        #     file_id+=1
        #     continue
        file_path = os.path.join(dataset_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
        for column_id, column in enumerate(header, start=1):
            new_id = f"{file_id}_{column_id}"
            all_docs_id.append(new_id)

            print(f"Processing file: {file_path} {file_id} {column_id}")
            
            #actual dataset jaccard similarity
            # start_time = time.time()
            # actual_jaccard = actual_jaccard_similarity(query_data, set_b)
            # end_time = time.time()
            # dataset_time += end_time - start_time
            # with open(output_actual_doc_id_file, 'r') as f:
            #     actual_doc_id = [line.strip() for line in f if line.strip()]
            # if actual_jaccard >= jaccard_threshold:
            #     actual_doc_id.append(new_id)
            #     with open(output_actual_doc_id_file, 'a') as f:
            #         f.write(f"{new_go get github.com/ekzhu/lshensembleid}\n")

            # cms sketch jaccard similarity
            # read cms sketch from file
            cms_b = CountMinSketch(width, depth)
            file_safe_name = os.path.splitext(os.path.basename(file))[0]
            input_filename = f"{file_safe_name}_{file_id}_{column_id}.txt"
            input_path = os.path.join(input_folder_cms, input_filename)
            print("input path")
            print(input_path)

            with open(input_path, 'r') as f:
                for i, line in enumerate(f):
                    cms_b.table[i] = np.array(list(map(int, line.strip().split())))
            print(input_filename)
            
            input_path = os.path.join(input_folder_minhash, input_filename)
            signature_b = load_tmh_txt(input_path,"ids")
            # estimate cms minhash jaccard similarity
            print("sig1 length:", len(query_signature))
            print("sig2 length:", len(signature_b))
            if len(query_signature) != len(signature_b):
                print("signature length not matched")
                with open(failed_files, "a") as f:
                    f.write(f"{query_name}.{query_column}.{query_selectivity}\n")
            # start_time = time.time()
            # cms_minhash_jaccard = cms_minhash_jaccard_similarity(query_signature, signature_b)
            # end_time = time.time()
            # cms_minhash_time += end_time - start_time
            # if cms_minhash_jaccard >= jaccard_threshold:
            #     cms_minhash_doc_id.append(new_id)
            
            # cms minhash lsh jaccard similarity
            minhash_signatures[new_id] = signature_b
        file_id += 1
    
    num_bands = find_optimal_bands(total_min_hash, jaccard_threshold, user_probability_lsh)
    print(f"Optimal number of bands: {num_bands}")
    lsh_index = build_lsh_index(minhash_signatures, num_bands)

    start_time = time.time()
    cms_minhash_lsh_candidates = find_similar_signatures(query_signature, num_bands, lsh_index)
    end_time = time.time()
    cms_minhash_lsh_time += end_time - start_time

    # parameter for final estimation
    alpha = 0
    count=0
    column_mappings="/home/sadiya/datadiscovery/nyc_column_mappings2.txt"
    for id in cms_minhash_lsh_candidates:
        count+=1
        # print(f"number of candidates: {len(cms_minhash_lsh_candidates)} count: {count} id: {id}")
        # action_type.csv.parent_type -> 1_3 here we are trying to get the file name from the column mappings file based on the id

        with open(column_mappings, 'r') as f:
            for line in f:
                print(line)
                column_name , column_id=line.split(" -> ")

                if column_id.strip() == id:

                    file,col=column_name.split(".csv.")
                    cms_filename=f"{file}_{id}.txt"
                    break
        print(cms_filename)
        cms_b = CountMinSketch(width, depth)
        input_path = os.path.join(input_folder_cms, cms_filename)
        with open(input_path, 'r') as f:
                for i, line in enumerate(f):
                    cms_b.table[i] = np.array(list(map(int, line.strip().split())))
        

        # with open(output_cms_minhash_lsh_candidates_doc_id_file, 'a') as f:
        #     f.write(f"{id}\n")
        start_time = time.time()
        cms_jaccard = cms_jaccard_similarity(query_cms, cms_b, depth, width)
        end_time = time.time()
        cms_minhash_lsh_time += end_time - start_time
        if cms_jaccard >= jaccard_threshold:
            cms_minhash_lsh_doc_id.append(id)
            # with open(output_cms_min pnyhash_lsh_doc_id_file, 'a') as f:
            #             f.write(f"{id}\n")
    # print("for looop finished")
                        
    # time.sleep(500)
    metrics_file = f"nyc_tmh_Run{args.run}.csv"
    actual_set = set(actual_doc_id)
    all_doc_ids = set(all_docs_id)
    print("###################################### cms_minhash_lsh")
    print("time taken by cms_minhash_lsh:", cms_minhash_lsh_time+query_minhash_creation_time)
    estimated_set = set(cms_minhash_lsh_doc_id)
    TP = len(estimated_set & actual_set)
    FP = len(estimated_set - actual_set)
    FN = len(actual_set - estimated_set)
    TN = len(all_doc_ids - (estimated_set | actual_set))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0

    print(f"True Positives (TP): {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Negatives (TN): {TN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    row_cms_minhash_lsh = {
        "method_name": "cms_minhash_lsh",
        "query_file": query_file,     # assuming query_file is defined
        "query_column": query_column,
        "query_selectivity": query_selectivity,  # assuming selectivity is defined
        "TP":TP,
        "FP":FP,
        "FN":FN,
        "TN":TN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "time": cms_minhash_lsh_time,
        "minhash_creation_time" : query_minhash_creation_time,
        "threshold": jaccard_threshold,       
        "dataset": dataset_name,
        "width": width,
        "depth": depth,
        "candidates":len(cms_minhash_lsh_candidates)
    }

    with open(metrics_file, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row_cms_minhash_lsh.keys())
        # writer.writerow(row_cms)
        # writer.writerow(row_cms_sampling)
        # writer.writerow(row_cms_earlystopping)
        # writer.writerow(row_cms_minhash)
        # writer.writeheader()  # Write header only once
        writer.writerow(row_cms_minhash_lsh)
    print(f"Metrics written to {metrics_file}")
    