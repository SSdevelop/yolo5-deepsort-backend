import json
import os
import sys

def top_k_results(data, k:int, run_type:str):
    if run_type == 'image':
        all_results = []

        for frame in data['result']:
            if 'scores' in frame:
                for score, box in zip(frame['scores'], frame['boxes']):
                    all_results.append((frame['frame'], score, box))

        all_results.sort(key=lambda x: x[1], reverse=True)
        top_k = all_results[:k]

        new_data = [{'frame': i} for i in range(max([frame for frame, _, _ in all_results]) + 1)]

        for frame, score, box in top_k:
            if 'scores' not in new_data[frame]:
                new_data[frame]['scores'] = []
                new_data[frame]['boxes'] = []

            new_data[frame]['scores'].append(score)
            new_data[frame]['boxes'].append(box)
    elif run_type == 'lang':
        all_results = {}

        for frame in data['result']:
            if 'scores' in frame:
                for score, box, label in zip(frame['scores'], frame['boxes'], frame['labels']):
                    if label not in all_results:
                        all_results[label] = []
                    all_results[label].append((frame['frame'], score, box))

        for label, results in all_results.items():
            results.sort(key=lambda x: x[1], reverse=True)
            all_results[label] = results[:k]

        max_frame = max([frame for results in all_results.values() for frame, _, _ in results])
        new_data = [{'frame': i} for i in range(max_frame + 1)]

        for label, results in all_results.items():
            for frame, score, box in results:
                if 'scores' not in new_data[frame]:
                    new_data[frame]['scores'] = []
                    new_data[frame]['boxes'] = []
                    new_data[frame]['labels'] = []
                new_data[frame]['scores'].append(score)
                new_data[frame]['boxes'].append(box)
                new_data[frame]['labels'].append(label)
    else:
        print("Invalid run_type: must be image or lang ")

    return new_data

if __name__ == "__main__":
    if len(sys.argv)!=2:
        print('Usage: python topK_analysis.py results/{json_file.json}')
    #input_file = 'results/IMG_1752.mp4_lang.json'  # name of the file to be processed
    input_file=sys.argv[1]
    with open(input_file, 'r') as f:
        data = json.load(f)

    k = 20  # set a k
    # choose the type of query（img/lang）
    #type = 'image'  # Image
    type = 'lang'   # Language
    new_result = top_k_results(data, k, type)

    data['result'] = new_result

    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_topk.json"

    with open(output_file, 'w') as f:
        json.dump(data, f)

    print("done")
