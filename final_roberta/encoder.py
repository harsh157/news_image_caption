
from resnet import resnet152
import spacy
import numpy as np
import torch
import time
import os
import pickle as pkl
from preprocess import sen_embed
from joblib import parallel_backend
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load(i,context, metadata, nlp):
    c = context[i]
    article_id = metadata[i]['article_id']
    embed_path = f'/home/jupyter/data/GoodNews/sen_embeddings/{article_id}.pkl'
    #print(embed_path)
    if os.path.exists(embed_path):
        try:
            with open(embed_path, 'rb') as pfile:
                v,vl = pkl.load(pfile)
        except:
            c_doc = nlp(c.lower())
            #print(c_doc)
            v = [token.vector for token in c_doc if token.has_vector]
            vl = len(v)
            #print(vl)
            with open(embed_path, 'wb') as pfile:
                pkl.dump((v,vl),pfile)
    
    else:
        c_doc = nlp(c.lower())
        #print(c_doc)
        v = [token.vector for token in c_doc if token.has_vector]
        vl = len(v)
        #print(vl)
        with open(embed_path, 'wb') as pfile:
            pkl.dump((v,vl),pfile)
    v = v[:300]
    vl = len(v)
    return v,vl
 

def load_roberta(i,context, metadata, roberta):
    c = context[i]
    article_id = metadata[i]['article_id']
    embed_path = f'/mnt/disks/embeds/GoodNews/roberta_embeddings/{article_id}.pkl'
    #print(embed_path)
    tokens = roberta.encode(c)[:512]
    pad_x = torch.ones((512), dtype=tokens.dtype)
    pad_x[:tokens.size(0)] = tokens
    if os.path.exists(embed_path):
        try:
            with open(embed_path, 'rb') as pfile:
                features = pkl.load(pfile)
                if len(features.size()) == 3:
                    features = features[0]
            #print(features.size(0))
            if features.size(0) == 512:
                return features, pad_x
        except:
            pass 
    with torch.no_grad():
        features = roberta.extract_features(pad_x)[0]
    print("RE-----",features.size())
    with open(embed_path, 'wb') as pfile:
        pkl.dump(features,pfile)
    return features, pad_x
 

class Encoder:
    def __init__(self):
        self.resnet = resnet152()
        self.resnet.to(device)
        #self.nlp = spacy.load('en_core_web_lg',disable=['textcat', 'parser', 'tagger', 'ner'])
        self.roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.roberta.to(device)
        self.roberta.eval()
        self.resnet.eval()

    def forward(self, image, context, metadata):
        image = image.to(device)
        start = time.time()
        with torch.no_grad():
            X_image = self.resnet(image)
        #print('Resnet time:', time.time()-start)
        X_image = X_image.permute(0, 2, 3, 1)
        B, H, W, C = X_image.shape
        P = H * W
        X_image = X_image.view(B, P, C)
        #print('Resnet time:', time.time()-start)


        #start_nlp = time.time()
        vs = []
        v_lens = []

        #with parallel_backend('threading', n_jobs=4):
        #    out = Parallel()(delayed(load)(i,context,metadata,self.nlp) for i in range(len(context)))
        #    for v,vl in out:
        #        vs.append(np.array(v))
        #        v_lens.append(vl)
                
        #for i,c in enumerate(context):
        #    article_id = metadata[i]['article_id']
        #    embed_path = f'/home/jupyter/data/GoodNews/sen_embeddings/{article_id}.pkl'
        #    #print(embed_path)
        #    if os.path.exists(embed_path):
        #        with open(embed_path, 'rb') as pfile:
        #            v,vl = pkl.load(pfile)

        #    else:
        #        c_doc = self.nlp.pipe([c.lower()])
        #        v = [token.vector for token in doc if token.has_vector]
        #        vl = len(v)
        #    vs.append(np.array(v))
        #    v_lens.append(vl)



        #context = [c.lower() for c in context]
        #context_docs = sen_embed.nlp.pipe(context, n_process=4)
        #for doc in context_docs:
        #    v = [token.vector for token in doc if token.has_vector]
        #    v_lens.append(len(v))
        #    vs.append(np.array(v))
        #for v,vl in context:
        #    vs.append(np.array(v))
        #    v_lens.append(vl)
        #max_len = max(v_lens)
        #context_vector = X_image.new_full((B, max_len, 300), np.nan)
        #for i, v in enumerate(vs):
        #    v_len = v.shape[0]
        #    v_tensor = torch.from_numpy(v).type_as(context_vector)
        #    context_vector[i, :v_len] = v_tensor

        #tokens = []
        #for c in context:
        #    tokens.append(self.roberta.encode(c)[:512])

        #X_article_ids = pad_sequence(tokens,padding_value=1,batch_first=True)
        #print('Tokens time:', time.time()-start)
        ##print(X_article_ids.size())
        ##X_article_ids = torch.stack(X_article_ids)
        #article_padding_mask = X_article_ids == 1
        #article_padding_mask = article_padding_mask.unsqueeze(1)
        #with torch.no_grad():
        #    X_article = self.roberta.extract_features(X_article_ids.cuda())

        X_article_ids = []
        X_article = []
        with parallel_backend('threading', n_jobs=4):
            out = Parallel()(delayed(load_roberta)(i,context,metadata,self.roberta) for i in range(len(context)))
            for features,tokens in out:
                X_article.append(features)
                X_article_ids.append(tokens)

        X_article_ids = torch.stack(X_article_ids, dim=0)
        X_article = torch.stack(X_article, dim=0)
        article_padding_mask = X_article_ids == 1
        article_padding_mask = article_padding_mask.unsqueeze(1)
        #print('Roberta time:', time.time()-start)
        #article_padding_mask = torch.isnan(context_vector).any(dim=-1)
        #X_article = context_vector
        #X_article[article_padding_mask] = 0
        image_padding_mask = X_image.new_zeros(B, P).bool()
        context = {
                'image': X_image.to(device),
                'image_mask': image_padding_mask.to(device),
                'article': X_article.to(device),
                'article_mask': article_padding_mask.to(device),
                }
        #print('Total time:', time.time()-start)

        return context
