import matplotlib.pyplot as plt
import json
import numpy as np

if __name__=='__main__':
    with open('data/hong_kong_airport_demo_data.mp4_scores.json','r') as f:
        results=json.load(f)
    print(results)
    results_np={ name:np.array(logits) for name,logits in results['result'].items()}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for name,result in results_np.items():
        ax.plot(result,label=name)
    plt.legend(loc="upper left")
    plt.ylim([0, 35])
    plt.ylabel('Cosine similarity between each language query and  the image')
    plt.xlabel('Frame sampled at 10fps')
    plt.savefig('results.jpg')