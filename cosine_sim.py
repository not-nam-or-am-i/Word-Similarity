import json
from scipy import spatial
from scipy import stats
import numpy as np

#save word2vec vào 1 dictionary, lưu dictionary đó vào 1 json file
def save_emb(inputfile, outputfile):
    #save word2vec vào dictionary
    embeddings = {}

    with open(inputfile, 'r', encoding='utf8') as f:
        all_lines = f.readlines()
        
        for i in range(2, len(all_lines)):
            words = all_lines[i].split()
            vec = []

            for j in range (1, len(words)):
                vec.append(float(words[j]))

            embeddings[words[0]] = vec

    #save dictionary vào json file
    with open(outputfile, 'w') as fp:
        json.dump(embeddings, fp)

    return

#load embeddings từ file json
def load_emb(filename):
    with open(filename, 'r') as fp:
        embeddings = json.load(fp)
    return embeddings

#tính cosin similarity
#xuất hiện từ trong visim-400 không nằm trong W2V_150 (cõi_tục, ...)
def cos_sim(embeddings):
    #read visim
    with open('datasets/ViSim-400/Visim-400.txt', 'r', encoding='utf8') as fp:
        visim = fp.readlines()

    #---------------------------------------------------------

    #cosine similarity
    v = []
    list1 = []
    list2 = []
    cosine_similarity = []
    
    for i in range(1, len(visim)):
        s = visim[i].split()
        u1 = s[0].strip()
        u2 = s[1].strip()
        if not(u1 in embeddings):
            continue
        if not(u2 in embeddings):
            continue

        v1 = embeddings[u1]
        v2 = embeddings[u2]

        sim = 1 - spatial.distance.cosine(v1, v2)
        cosine_similarity.append(sim)
        list1.append(u1)
        list2.append(u2)
        v.append(float(s[3].strip()))

    #---------------------------------------------------------

    #chuyển từ [-1, 1] sang [0, 6]
    arr = np.array(cosine_similarity)
    arr = (arr+1) * 3
    cosine_similarity = arr.tolist()

    #---------------------------------------------------------

    print('----------------------------------------------------------------------')
    print('Cosine similarity cho các cặp từ trong visim-400:')
    print(cosine_similarity)
    print('----------------------------------------------------------------------')

    #---------------------------------------------------------
    #correlation
    print(" Pearson correlation coefficient: ", stats.pearsonr(cosine_similarity,v))
    print(" Spearman's rank correlation coefficient: ", stats.spearmanr(cosine_similarity,v))

    #---------------------------------------------------------
    #export ra file txt
    with open('cosine_similarity.txt', 'w', encoding='utf8') as f:
        for i in range(len(cosine_similarity)):
            f.write(list1[i] + ' ' + list2[i] + ' ' + str(cosine_similarity[i]) +'\n')

    return cosine_similarity

def main():
    save_emb('W2V_150.txt', 'embeddings.json')
    embeddings = load_emb('embeddings.json')
    cosine_similarity = cos_sim(embeddings)

    return

if __name__ == '__main__':
    main()