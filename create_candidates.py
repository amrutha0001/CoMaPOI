"""
Create candidate POI files for RAG (Retrieval-Augmented Generation)
Run this before running inference_inverse_new.py
"""
import json
import random
import os
import argparse

def create_candidates(dataset='nyc', num_users=500):
    """
    Create candidates file for a dataset
    
    Args:
        dataset: Dataset name (nyc, tky, ca)
        num_users: Number of users to create candidates for
    """
    # Dataset-specific max POI IDs
    max_poi_ids = {
        'nyc': 5091,
        'tky': 7851,
        'ca': 13630
    }
    
    max_poi_id = max_poi_ids.get(dataset, 5091)
    
    # Create directory
    os.makedirs(f'dataset_all/{dataset}/train', exist_ok=True)
    candidates_file = f'dataset_all/{dataset}/train/{dataset}_train_candidates.jsonl'
    
    print(f"📝 Creating {candidates_file}...")
    print(f"   Dataset: {dataset}")
    print(f"   Users: {num_users}")
    print(f"   Max POI ID: {max_poi_id}")
    
    # Create candidates
    with open(candidates_file, 'w') as f:
        for user_id in range(num_users):
            # Generate 100 random POI IDs as candidates
            candidates = random.sample(range(1, max_poi_id + 1), 100)
            
            # Write to file
            f.write(json.dumps({
                'user_id': str(user_id),
                'candidates': candidates
            }) + '\n')
    
    print(f"✅ Created {candidates_file}")
    print(f"   Total entries: {num_users}")
    
    return candidates_file

def main():
    parser = argparse.ArgumentParser(description="Create candidate POI files")
    parser.add_argument('--dataset', type=str, default='nyc', 
                       choices=['nyc', 'tky', 'ca'],
                       help='Dataset name')
    parser.add_argument('--num_users', type=int, default=500,
                       help='Number of users to create candidates for')
    
    args = parser.parse_args()
    
    create_candidates(args.dataset, args.num_users)
    print("\n🎉 Done! You can now run inference_inverse_new.py")

if __name__ == "__main__":
    main()