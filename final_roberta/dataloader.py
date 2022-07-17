
from torch import Tensor
import pickle as pkl
from pymongo import MongoClient
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple,List
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
import pymongo
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os
import glob
import preprocess
import spacy
import time
from transformers import BertTokenizerFast

client = MongoClient(host='localhost', port=27017)
PAD_IDX = 0
MAX_LENGTH = 512

class GoodNewsVocab:
  def __init__(self):
    self.word2idx = {'<s>':1,'</s>':2, '<unk>':3}
    self.idxtoword = {1:'<s>', 2:'</s>', 3:'<unk>'}
    self.unk_words = []
    self.max_idx = 2
    self.pad_idx = 0
  
  def get_idx(self, word):
    if word in self.unk_words:
      return self.word2idx['<unk>']
    if word not in self.word2idx:
      self.word2idx[word] = self.max_idx+1
      self.max_idx += 1
      self.idxtoword[self.word2idx[word]] = word
    return self.word2idx[word]
  
  def process_caption(self, caption):
    caption_token = []
    for token in caption:
      caption_token.append(self.get_idx(token))
    return caption_token


  def count_all_words(self, all_captions, count_threshold=4):
    count_word = {}
    for caption in all_captions:
      tokens = caption
      for tok in tokens:
        count_word[tok] = count_word.get(tok,0) + 1
    for w in count_word:
      if count_word[w] <= count_threshold:
        self.unk_words.append(w)
    print(self.unk_words)
    print(count_word)
    

class SentenceEmbed:
  def __init__(self):
    self.nlp = spacy.load('en_core_web_lg',
                              disable=['textcat', 'parser', 'tagger', 'ner'])
    
  def embed(self, context):
    context = context.lower()
    doc = self.nlp.pipe([context])
    #print(doc)
    iter = 0
    for d in doc:
      iter += 1
      v = [token.vector for token in d if token.has_vector]
    vlen = len(v)

    return v,vlen



class GoodNewsDataset:

    def __init__(self, split='train', tok_caption_path='', start_idx = 0, _max_len=512):
        self.split = split
        with open('/home/jupyter/tokenized_caption.pkl','rb') as pfile:
            self.tokenized_caption = pkl.load(pfile)
        with open('/home/jupyter/vocab.pkl', 'rb') as vfile:
            self.vocab = pkl.load(vfile)
        self.image_dir = '/home/jupyter/data/GoodNews/goodnews/images_processed'
        self.preprocess = Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.db = client.goodnews
        sample_cursor = client.goodnews.splits.find({'split': {'$eq': split}}, projection=['_id']).sort('_id', pymongo.ASCENDING)
        all_images = glob.glob('/home/jupyter/data/GoodNews/goodnews/images_processed/*jpg')
        all_images = set([image.split('/')[-1].split('.')[0] for image in all_images])
        print(client.goodnews.splits.count_documents({'split': {'$eq': split}}))
        self.ids = np.array([article['_id'] for article in tqdm(sample_cursor) if article['_id'] in all_images])
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        print(len(self.ids))
        sample_cursor.close()
        self.start_idx = start_idx
        #self.embedder = SentenceEmbed()

    def __getitem__(self, idx):
        idx += self.start_idx
        idx = idx % (len(self.ids))
        sample_id = self.ids[idx]
        sample = self.db.splits.find_one({'_id': {'$eq': sample_id}})
        article = self.db.articles.find_one({'_id': {'$eq': sample['article_id']},}, projection=['_id', 'context', 'images', 'web_url'])
        image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
        image = Image.open(image_path)
        image_index = sample['image_index']
        caption = article['images'][image_index]
        context = ' '.join(article['context'].strip().split(' ')[:500])
        #caption = article['images'][image_index]
        #embed_path = f'/home/jupyter/data/GoodNews/sen_embeddings/{sample["article_id"]}.pkl'
        #if os.path.exists(embed_path):
        #    with open(embed_path, 'rb') as pfile:
        #        context = pkl.load(pfile)
        #else:
        #    context = self.embedder.embed(context)


        #if sample_id in self.tokenized_caption:
        #    caption_tokens = self.tokenized_caption[sample_id]
        #else:
        #    caption_tokens = self.tokenize(caption)
        #    self.tokenized_caption[sample_id] = caption_tokens
        caption_tokens = self.tokenizer.encode(caption)[:MAX_LENGTH]


        #caption = ' '.join(caption_tokens)
        #print(caption)
        #caption_tokens = self.prepare_target_tokens(caption_tokens)
        
        metadata = {'web_url': article['web_url'], 'image_path': image_path, 'caption':caption.strip(), 'article_id': sample['article_id']}
        image = self.preprocess(image)
        return np.array(caption_tokens,dtype=np.int),image,context,metadata

    def tokenize(self, caption):
        processed = preprocess.preprocess_sentence(caption)
        template, full = preprocess.NER(' '.join(processed))
        template = [token if '_' in token else token.lower() for token in template]
        return template


    def prepare_target_tokens(self, caption):
        caption = [self.vocab['word2idx'][token] if token in self.vocab['word2idx'] else self.vocab['word2idx']['<unk>'] for token in caption]
        return [self.vocab['word2idx']['<s>']] + caption + [self.vocab['word2idx']['</s>']]

    def __len__(self):
        return len(self.ids)

def create_masks(src_batch: Tensor, tgt_batch: Tensor) -> Tuple[Tensor, Tensor]:
    # ----------------------------------------------------------------------
    # [1] padding mask
    # ----------------------------------------------------------------------
    
    # (batch_size, 1, max_tgt_seq_len)
    src_pad_mask = (src_batch != PAD_IDX).unsqueeze(1)
    
    # (batch_size, 1, max_src_seq_len)
    tgt_pad_mask = (tgt_batch != PAD_IDX).unsqueeze(1)

    # ----------------------------------------------------------------------
    # [2] subsequent mask for decoder inputs
    # ----------------------------------------------------------------------
    max_tgt_sequence_length = tgt_batch.shape[1]
    tgt_attention_square = (max_tgt_sequence_length, max_tgt_sequence_length)

    # full attention
    full_mask = torch.full(tgt_attention_square, 1)
    
    # subsequent sequence should be invisible to each token position
    subsequent_mask = torch.tril(full_mask)
    
    # add a batch dim (1, max_tgt_seq_len, max_tgt_seq_len)
    subsequent_mask = subsequent_mask.unsqueeze(0)

    return src_pad_mask, tgt_pad_mask & subsequent_mask



def collate_fn(batch):
    start = time.time()
    target_tokens_list = []
    image_batch = []
    context_batch = []
    ntokens = 0
    metadata_batch = []
    for i, (caption, image, context, metadata) in enumerate(batch):
        # Tokenization
        target_tokens_list.append(torch.from_numpy(caption) )
        ntokens += len(caption) -1
        image_batch.append(image)
        context_batch.append(context)
        metadata_batch.append(metadata)

    image_batch = torch.stack(image_batch)
    target_batch = pad_sequence(target_tokens_list,padding_value=0,batch_first=True)
    label_batch  = target_batch[:, 1:]
    target_batch = target_batch[:, :-1]
    source_mask, target_mask = create_masks(target_batch, target_batch)
    #print('Batch Load:',time.time()-start)
    return [target_batch, target_mask, label_batch, image_batch, context_batch, ntokens, metadata_batch]

def make_loader(batch_size=16):
    dataset = GoodNewsDataset()
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
#dataset = GoodNewsDataset()
#DataLoader(dataset, batch_size = 16, collate_fn=collate_fn)
