import math
import json
import csv

def evaluate_poi_predictions(args, file_path, top_k, output_file, csv_file, key):
    """
    Evaluate prediction results in the output JSON file, calculate multiple metrics, and selectively output metrics based on top_k.
    Supports top_k values of 1, 5, 10, 20, calculating metrics such as ACC, MRR, and NDCG.
    
    Args:
        args: Command line arguments
        file_path: Path to the JSON file containing prediction results
        top_k: Length of the predicted POI ID list, supports 1, 5, 10, 20
        output_file: Path to the txt file to save evaluation results
        csv_file: Path to the csv file to save evaluation results
        key: Key in the JSON file containing predicted POI IDs
        
    Returns:
        metrics: A dictionary containing the calculated results
    """
    total_samples = 0
    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_20 = 0
    reciprocal_rank_sum = 0.0
    ndcg_sum_5 = ndcg_sum_10 = ndcg_sum_20 = 0.0

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for sample in data:
        total_samples += 1
        true_poi_id = str(sample.get('label'))
        predicted_poi_ids = sample.get(key, [])

        predicted_poi_ids = [str(poi_id) for poi_id in predicted_poi_ids]  # Convert to strings

        # Calculate ACC@1, @5, @10, @20
        if predicted_poi_ids and predicted_poi_ids[0] == true_poi_id:
            hit_at_1 += 1
        if true_poi_id in predicted_poi_ids[:5]:
            hit_at_5 += 1
        if true_poi_id in predicted_poi_ids[:10]:
            hit_at_10 += 1
        if true_poi_id in predicted_poi_ids[:20]:
            hit_at_20 += 1

        # Calculate Reciprocal Rank (MRR)
        try:
            rank = predicted_poi_ids.index(true_poi_id) + 1
            reciprocal_rank_sum += 1 / rank
        except ValueError:
            pass  # true_poi_id not in prediction list

        # Calculate NDCG@5, @10, @20
        def calculate_dcg(rel_list, k):
            dcg = 0
            for i in range(min(k, len(rel_list))):
                rel = rel_list[i]
                dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed and log base is 2
            return dcg

        def calculate_ndcg(rel_list, k):
            dcg = calculate_dcg(rel_list, k)
            ideal_rel_list = sorted(rel_list, reverse=True)
            idcg = calculate_dcg(ideal_rel_list, k)
            return dcg / idcg if idcg > 0 else 0

        # Create relevance list (1 for true_poi_id, 0 for others)
        rel_list_5 = [1 if pid == true_poi_id else 0 for pid in predicted_poi_ids[:5]]
        rel_list_10 = [1 if pid == true_poi_id else 0 for pid in predicted_poi_ids[:10]]
        rel_list_20 = [1 if pid == true_poi_id else 0 for pid in predicted_poi_ids[:20]]

        # Calculate NDCG for different k values
        ndcg_5 = calculate_ndcg(rel_list_5, 5)
        ndcg_10 = calculate_ndcg(rel_list_10, 10)
        ndcg_20 = calculate_ndcg(rel_list_20, 20)

        ndcg_sum_5 += ndcg_5
        ndcg_sum_10 += ndcg_10
        ndcg_sum_20 += ndcg_20

    # Calculate final metrics
    acc_at_1 = hit_at_1 / total_samples * 100 if total_samples > 0 else 0
    acc_at_5 = hit_at_5 / total_samples * 100 if total_samples > 0 else 0
    acc_at_10 = hit_at_10 / total_samples * 100 if total_samples > 0 else 0
    acc_at_20 = hit_at_20 / total_samples * 100 if total_samples > 0 else 0

    mrr = reciprocal_rank_sum / total_samples * 100 if total_samples > 0 else 0

    ndcg_at_5 = ndcg_sum_5 / total_samples * 100 if total_samples > 0 else 0
    ndcg_at_10 = ndcg_sum_10 / total_samples * 100 if total_samples > 0 else 0
    ndcg_at_20 = ndcg_sum_20 / total_samples * 100 if total_samples > 0 else 0

    # Create metrics dictionary
    metrics = {
        'total_samples': total_samples,
        'HR@1': acc_at_1,
        'HR@5': acc_at_5,
        'HR@10': acc_at_10,
        'HR@20': acc_at_20,
        'MRR': mrr,
        'NDCG@5': ndcg_at_5,
        'NDCG@10': ndcg_at_10,
        'NDCG@20': ndcg_at_20
    }

    # Write metrics to txt file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"HR@1: {acc_at_1:.2f}\n")
        f.write(f"HR@5: {acc_at_5:.2f}\n")
        f.write(f"HR@10: {acc_at_10:.2f}\n")
        f.write(f"HR@20: {acc_at_20:.2f}\n")
        f.write(f"MRR: {mrr:.2f}\n")
        f.write(f"NDCG@5: {ndcg_at_5:.2f}\n")
        f.write(f"NDCG@10: {ndcg_at_10:.2f}\n")
        f.write(f"NDCG@20: {ndcg_at_20:.2f}\n")

    # Write metrics to csv file
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total samples', total_samples])
        writer.writerow(['HR@1', f"{acc_at_1:.2f}"])
        writer.writerow(['HR@5', f"{acc_at_5:.2f}"])
        writer.writerow(['HR@10', f"{acc_at_10:.2f}"])
        writer.writerow(['HR@20', f"{acc_at_20:.2f}"])
        writer.writerow(['MRR', f"{mrr:.2f}"])
        writer.writerow(['NDCG@5', f"{ndcg_at_5:.2f}"])
        writer.writerow(['NDCG@10', f"{ndcg_at_10:.2f}"])
        writer.writerow(['NDCG@20', f"{ndcg_at_20:.2f}"])

    print(f"Evaluation results saved to {output_file} and {csv_file}")
    print(f"HR@{top_k}: {metrics[f'HR@{top_k}']:.2f}")
    print(f"MRR: {mrr:.2f}")
    print(f"NDCG@{top_k}: {metrics[f'NDCG@{top_k}']:.2f}")

    return metrics
