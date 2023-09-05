import json
import matplotlib.pyplot as plt
import sys
if __name__=='__main__':
    if len(sys.argv)!=2:
        print('Usage: python hist.py {scores.json}')
    with open(sys.argv[1]) as f:
        data = json.load(f)
    scores = []
    print(f"Size of {sys.argv[1]}: {len(data['result'])}")
    for result in data['result']:
        if 'logits' in result:
            scores.extend(result['logits'])
    plt.hist(scores, bins=50, color='b', edgecolor='black')
    plt.title('Scores Distribution ('+f.name+')')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    hist_name=f"{str(sys.argv[1])}_hist.jpg"
    plt.savefig(hist_name)