import heapq
from operator import itemgetter
from scipy import spatial
import json

#load embeddings từ file json
def load_emb(filename):
    with open(filename, 'r') as fp:
        embeddings = json.load(fp)
    return embeddings

#------------------------------------------------------
#cosine similarity
def cosine_similarity(embeddings, word):
    result = {}
    v1 = embeddings[word]

    for value in embeddings.keys():
        if value == word:
            continue

        v2 = embeddings[value]
        sim = 1 - spatial.distance.cosine(v1, v2)
        result[value] = sim

    return result
#------------------------------------------------------
def k_nearest_words(k, word, embeddings):
    if not(word in embeddings):
        print(word, 'not found!')
        return

    cos_sim = cosine_similarity(embeddings, word)
    topword = heapq.nlargest(k, cos_sim.items(), key=itemgetter(1))
    topworddict = dict(topword)
    
    return topworddict

def main():
    embeddings = load_emb('embeddings.json')
    kwords = k_nearest_words(5, 'của', embeddings)
    print(kwords)
    return

if __name__ == '__main__':
    main()