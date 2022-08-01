import logging
import os
import random
from typing import Dict

import numpy as np
import pymongo
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from pymongo import MongoClient
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
#import torch
import numpy as np
import pickle as pkl

from tqdm import tqdm

from tell.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('goodnews_entity')
class EntityGoodNewsReader(DatasetReader):
    """Read from the Good News dataset.

    See the repo README for more instruction on how to download the dataset.

    Parameters
    ----------
    tokenizer : ``Tokenizer``
        We use this ``Tokenizer`` for both the premise and the hypothesis.
        See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``
        We similarly use this for both the premise and the hypothesis.
        See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 entity_embed_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 eval_limit: int = 5120,
                 filter_entities_groups: bool = False,
                 entity_limit: int = 100,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.client = MongoClient(host=mongo_host, port=mongo_port)
        self.db = self.client.goodnews
        self.image_dir = image_dir
        self.entity_embed_dir = entity_embed_dir
        self.preprocess = Compose([
            # Resize(256), CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.eval_limit = eval_limit
        random.seed(1234)
        self.rs = np.random.RandomState(1234)
        self.filter_entities_groups = filter_entities_groups
        self.max_entity_size = entity_limit

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        # Setting the batch size is needed to avoid cursor timing out
        # We limit the validation set to 1000
        logger.info('Grabbing all article IDs')
        limit = self.eval_limit if split == 'val' else 0
        sample_cursor = self.db.splits.find({
            'split': {'$eq': split},
        }, projection=['_id'], limit=limit).sort('_id', pymongo.ASCENDING)
        ids = np.array([article['_id'] for article in tqdm(sample_cursor)])
        sample_cursor.close()
        self.rs.shuffle(ids)

        err = 0
        for sample_id in tqdm(ids):
            sample = self.db.splits.find_one({'_id': {'$eq': sample_id}})

            # Find the corresponding article
            article = self.db.articles.find_one({
                '_id': {'$eq': sample['article_id']},
            }, projection=['_id', 'context', 'images', 'web_url'])

            # Load the image
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            entity_path = os.path.join(self.entity_embed_dir, f"{sample['article_id']}.pkl")
            entity_vec_path = os.path.join(self.entity_embed_dir, f"{sample['article_id']}.npy")
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            try:
                with open(entity_path, 'rb') as efile:
                    entities = pkl.load(efile)
                with open(entity_vec_path, 'rb') as efile:
                    entities_vector = np.load(efile)
            except (FileNotFoundError, OSError):
                err += 1
                continue

            yield self.article_to_instance(article, image, sample['image_index'], image_path, entities, entities_vector)

    def article_to_instance(self, article, image, image_index, image_path, entities, entities_vector) -> Instance:
        context = ' '.join(article['context'].strip().split(' ')[:500])

        caption = article['images'][image_index]
        caption = caption.strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)
        ent_features, ent_bpe, ent_metadata = self.getEntityEmbed(entities, entities_vector) 
        ent_bpe = np.array(ent_bpe)
        if len(ent_features) == 0 or len(ent_bpe) == 0:
            ent_features = np.array([[]])
            ent_bpe = np.array([])
        #print(ent_features.shape)
        #print(ent_bpe.shape)

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
            'entity': ArrayField(ent_features, padding_value=np.nan),
            'entity_tokens': ArrayField(ent_bpe, padding_value=1)
        }

        metadata = {'context': context,
                    'caption': caption,
                    'web_url': article['web_url'],
                    'entity_tokens': ent_bpe,
                    'entity_metadata': ent_metadata,
                    'image_path': image_path}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def getEntityEmbed(self,ent_list, entities_vector):
        ent_features = []
        ent_bpe = []
        metadata = []
        ent_group = ['PERSON', 'NORP', 'ORG', 'DATE', 'TIME', 'FAC',
                'GPE', 'LOC', 'PRODUCT', 'EVENT', 'ART']
        idxs = []
        for ind, ent in enumerate(ent_list):
            if 'bpe_tok' not in ent:
                return np.array([]), [], []
            #if len(metadata) > self.max_entity_size:
            #    break
            if self.filter_entities_groups and ent['ent_type'] in ent_group:
                idxs.append(ind)
            #print(ent['ent_type'])
            vec = entities_vector[ind]
            #if len(vec) != 1024:
            #    continue
            ent_features.append(vec)
            ent_bpe.append(ent['bpe_tok'])
            metadata.append({'ent_type':ent['ent_type'], 'word': ent['word']})

        #if len(idxs) > 5:
        ent_features = [ent_features[idx] for idx in idxs]
        ent_bpe = [ent_bpe[idx] for idx in idxs]
        metadata = [metadata[idx] for idx in idxs]

        ent_features = ent_features[:self.max_entity_size]
        ent_bpe = ent_bpe[:self.max_entity_size]
        metadata = metadata[:self.max_entity_size]
        return np.array(ent_features), ent_bpe, metadata

