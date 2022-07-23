
import glob
import pickle as pkl
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

ent_0 = 0
ent_notok = 0
entlist = []

def process(fname):
    ent_0 = 0
    ent_notok = 0
    entlist = []
    with open(fname, 'rb') as pfile:
        ents = pkl.load(pfile)
    if len(ents) == 0:
        ent_0 += 1
    for ent in ents:
        if 'bpe_tok' not in ent:
            ent_notok += 1
            artid = fname.split('/')[-1].split('.')[0]
            print(artid)
            entlist.append(artid)
            break
    return ent_0, ent_notok, entlist


flist = glob.glob("/home/jupyter/data/GoodNews/entity_embeddings/embeddings/*pkl")
with parallel_backend('threading', n_jobs=4):
    for e0, et, el in Parallel()(delayed(process)(i) for i in tqdm(flist)):
        ent_0 += e0
        ent_notok += et
        entlist += el
print(ent_0)
print(ent_notok)
print(entlist)
