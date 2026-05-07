import sys
from embedding import bert_embedding
from sklearn.preprocessing import normalize
import numpy as np

from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.tree import nj

f = open(sys.argv[1])

seqs = []
names = []
x = f.readline()

while len(x) > 0:
    y = x.strip().split( )
    names.append(y[0][1:])
    x = f.readline().strip()
    seqs.append(x)

    x = f.readline()

def generate_tree_from_embeddings(embeddings, names, output_file="llm_tree.nwk"):

    print(f"Calculating distances for {len(names)} sequences...")
    
    dist_vector = pdist(embeddings, metric='euclidean')
    dist_matrix = squareform(dist_vector)
    dm = DistanceMatrix(dist_matrix, names)
    
    print("Constructing Neighbor-Joining tree...")
    tree = nj(dm)
    
    tree.write(output_file)
    print(f"Tree saved to {output_file}")
    
emb = bert_embedding(seqs, sys.argv[2])
emb = normalize(emb, norm='l2')
generate_tree_from_embeddings(emb, names, sys.argv[3])