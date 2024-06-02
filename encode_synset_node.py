import nltk
import networkx as nx
from nltk.corpus import wordnet as wn
from node2vec import Node2Vec
from tqdm import tqdm
import pandas
import ast

G = nx.DiGraph()
all_synsets = list(wn.all_synsets()) # len: 117659

have_edge_synsets = set()
edge_list = []
for synset in all_synsets:
    synset_name = str(synset)[8:-2]
    G.add_node(synset_name)
    
    for hyponym in synset.hyponyms():  # 17%的synset有hyponym, 比较稀疏
        hyponym_name = str(hyponym)[8:-2]
        
        have_edge_synsets.add(synset_name)
        have_edge_synsets.add(hyponym_name)
        
        G.add_edge(synset_name, hyponym_name)
        edge_list.append((synset_name, hyponym_name))
count = 0
for synset in all_synsets:
    synset_name = str(synset)[8:-2]
    if synset_name not in have_edge_synsets:
        count += 1

print ("count: ", count)
print ("len(all_synsets): ", len(all_synsets))
print ("ratio: ", count*1.0/len(all_synsets))


    
# node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=15, p=1, q=1)
# model = node2vec.fit(window=10, min_count=1, sg=1, epochs=25)

# all_synset_names = [str(synset)[8:-2] for synset in all_synsets]
# print ("all_synset_names: ", len(all_synset_names))
# print ("all_synset_names: ", all_synset_names[:10])

# data = pandas.read_csv("./dataset/train_wsd.csv")
# count = 0
# for sense_keys in tqdm(data["sense_keys"]):
#     sense_keys = ast.literal_eval(sense_keys)
    
#     sense_keys = [str(wn.lemma_from_key(key).synset())[8:-2] for key in list(sense_keys)]
#     for sense_key in sense_keys:
#         if sense_key not in all_synset_names:
#             print (sense_key)
#             count += 1

# print ("count: ", count)
    

# synset_vector = model.wv['dog.n.01']
# print("The vector for 'dog.n.01' is:", synset_vector)

# model.save("./ckpt/synset_node2vec.model")
# model.wv.save_word2vec_format('./ckpt/synset_node2vec.model', binary=False)