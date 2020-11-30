from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import json

def load_trainset(embeddings):
    x_train, y_train = [], []
    with open('antonym-synonym set\Antonym_vietnamese.txt', 'r', encoding='utf8') as f:
        ant_pairs = f.readlines()
    with open('antonym-synonym set\Synonym_vietnamese.txt', 'r', encoding='utf8') as f:
        syn_pairs = f.readlines()

    #-----------------------------------------------------------
    #antonym data, y=0
    for pair in ant_pairs:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not(u1 in embeddings) or not(u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1+v2)
        y_train.append(0)

    #-----------------------------------------------------------
    #synomyn data, y=1
    for pair in syn_pairs:
        words = pair.split()
        u1 = words[0].strip()
        try:
            u2 = words[1].strip()       #có dòng chỉ có 1 từ
        except:
            continue
        else:
            u2 = u2

        if not(u1 in embeddings) or not(u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_train.append(v1+v2)
        y_train.append(1)

    return x_train, y_train

def load_testset(embeddings):
    x_test, y_test = [], []
    with open('datasets/ViCon-400/400_noun_pairs.txt', 'r', encoding='utf8') as f:
        noun_pairs = f.readlines()
    with open('datasets/ViCon-400/400_verb_pairs.txt', 'r', encoding='utf8') as f:
        verb_pairs = f.readlines()
    with open('datasets/ViCon-400/600_adj_pairs.txt', 'r', encoding='utf8') as f:
        adj_pairs = f.readlines()
    testset = noun_pairs[1:] + verb_pairs[1:] + adj_pairs[1:]

    for pair in testset:
        words = pair.split()
        u1 = words[0].strip()
        u2 = words[1].strip()
        if not(u1 in embeddings) or not(u2 in embeddings):
            continue
        v1 = embeddings[u1]
        v2 = embeddings[u2]
        x_test.append(v1+v2)
        if words[2] == 'ANT':
            y_test.append(0)
        else:
            y_test.append(1)
        
    return x_test, y_test

#load embeddings từ file json
def load_emb(filename):
    with open(filename, 'r') as fp:
        embeddings = json.load(fp)
    return embeddings

def main():
    #load embeddings, trainset, testset
    embeddings = load_emb('embeddings.json')
    x_train, y_train = load_trainset(embeddings)
    x_test, y_test = load_testset(embeddings)
    
    #logistic regression train
    model = LogisticRegression()
    model.fit(x_train, y_train)

    #test
    pred = model.predict(x_test)
    print('Precision score:', precision_score(y_test, pred))
    print('Recall score:', recall_score(y_test, pred))
    print('F1 score:', f1_score(y_test, pred))
    print('Accuracy:', accuracy_score(y_test, pred))
    return

if __name__ == '__main__':
    main()