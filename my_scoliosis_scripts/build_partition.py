import os
import json
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to the created silhouette dataset (OUTPUT of preprocess)")
    parser.add_argument('--original_input', type=str, required=True, help="Path to the original video dateset (to infer train/test split)")
    parser.add_argument('--output_json', type=str, default="dataset_partition.json")
    args = parser.parse_args()

    # 1. List all subjects in the formatted dataset
    # Structure: Root / SubjectID / ...
    subjects = [d for d in os.listdir(args.dataset_root) if os.path.isdir(os.path.join(args.dataset_root, d))]
    
    train_set = []
    test_set = []
    
    print(f"Found {len(subjects)} subjects in dataset.")
    
    # 2. Heuristic to decide split
    # Check if SubjectID string or the corresponding original file path contains "train" or "test"
    # This is tricky because SubjectID might not exist 1:1 in original folder names if we constructed it from filenames.
    
    # Alternative: Just look at the folders. 
    # If the user's original data was:
    #   Input/Train/Patient/Bob.mp4
    #   Input/Test/Healthy/Alice.mp4
    #
    # Then Preprocessor created:
    #   Output/Bob/...
    #   Output/Alice/...
    
    # We need to find "Bob" in Input/Train.
    
    for subj in subjects:
        # Search for this subject name in original input
        # Naive search: is 'subj' a substring of any file path in 'train' folder?
        
        # Checking Train
        found_in_train = False
        # We need to search efficiently.
        # Let's assume if we can find the exact string match in filenames?
        
        # Let's walk the original dirs once and build a map
        pass

    # Better approach: Build a map of Original Structure
    subject_split_map = {}
    
    # Walk Original Input
    for root, dirs, files in os.walk(args.original_input):
        lower_root = root.lower()
        # User structure: data -> train-data, val-data
        is_train = 'train' in lower_root
        is_val = 'val' in lower_root
        is_test = 'test' in lower_root
        
        current_split = None
        if is_train: current_split = 'TRAIN_SET'
        elif is_test or is_val: current_split = 'TEST_SET' # OpenGait usually merges Test/Val for evaluation or we can just put Val in Test
        
        if current_split:
            for f in files:
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    # ID = Filename without extension
                    sid = os.path.splitext(f)[0]
                    sid = sid.replace(" ", "_")
                    subject_split_map[sid] = current_split

    # Now assign
    count_train = 0
    count_test = 0
    
    for subj in subjects:
        if subj in subject_split_map:
            split = subject_split_map[subj]
            if split == 'TRAIN_SET':
                train_set.append(subj)
                count_train += 1
            else:
                test_set.append(subj)
                count_test += 1
        else:
            # Default to Train if unknown? Or Test?
            print(f"Warning: Subject {subj} not found in original splits lookup. Defaulting to TRAIN.")
            train_set.append(subj)
            count_train += 1

    partition = {
        "TRAIN_SET": sorted(list(set(train_set))),
        "TEST_SET": sorted(list(set(test_set)))
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(partition, f, indent=4)
        
    print(f"Partition saved to {args.output_json}")
    print(f"Train: {len(partition['TRAIN_SET'])}, Test: {len(partition['TEST_SET'])}")

if __name__ == "__main__":
    main()
