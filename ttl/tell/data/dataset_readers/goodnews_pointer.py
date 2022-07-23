import logging
import os
import random
from collections import OrderedDict
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
from tell.data.fields import CopyTextField, ImageField, ListTextField

from tell.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('goodnews_entity_pointer')
class EntityPointerGoodNewsReader(DatasetReader):
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
                 context_token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 entity_embed_dir: str,
                 mongo_host: str = 'localhost',
                 mongo_port: int = 27017,
                 eval_limit: int = 5120,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._context_token_indexers = context_token_indexers
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
            }, projection=['_id', 'context', 'images',
                          'web_url', 'caption_ner', 'context_ner',
                          'context_parts_of_speech', 'caption_parts_of_speech'])
            named_entities = sorted(self._get_named_entities(article))

            copy_infos = self._get_caption_names(
                article, sample['image_index'])

            self._process_copy_tokens(copy_infos, article)
            proper_infos = self._get_context_names(article)


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
                if len(entities) == 0:
                    continue
                if 'bpe_tok' not in entities[0]:
                    continue
                with open(entity_vec_path, 'rb') as efile:
                    entities_vector = np.load(efile)
            except (FileNotFoundError, OSError):
                err += 1
                continue

            yield self.article_to_instance(article, image, named_entities, sample['image_index'], image_path, entities, copy_infos, proper_infos, entities_vector)

    def article_to_instance(self, article, image, named_entities, image_index, image_path, entities, copy_infos, proper_infos, entities_vector) -> Instance:
        context = ' '.join(article['context'].strip().split(' ')[:500])

        caption = article['images'][image_index]
        caption = caption.strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)
        ent_bpe, ent_metadata = self.getEntityEmbed(entities)

        fields = {
            'context': TextField(context_tokens, self._context_token_indexers),
            'image': ImageField(image, self.preprocess),
            'caption': CopyTextField(caption_tokens, self._token_indexers, copy_infos, None, 'caption'),
            #'caption': TextField(caption_tokens, self._token_indexers),
            'entity': ArrayField(entities_vector, padding_value=np.nan),
            'entity_tokens': ArrayField(np.array(ent_bpe), padding_value=np.nan)
        }

        metadata = {'context': context,
                    'caption': caption,
                    'web_url': article['web_url'],
                    'entity_tokens': ent_bpe,
                    'entity_metadata': ent_metadata,
                    'image_path': image_path}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def getEntityEmbed(self,ent_list):
        ent_bpe = []
        metadata = []
        for ent in ent_list:
            #vec = ent['vector']
            #if len(vec) != 1024:
            #    continue
            #ent_features.append(ent['vector'])
            ent_bpe.append(ent['bpe_tok'])
            #if 'bpe_tok' in ent:
            #    ent_bpe.append(ent['bpe_tok'])
            #else:
            #    ent_bpe.append(1)
            metadata.append({'ent_type':ent['ent_type'], 'word': ent['word']})

        return ent_bpe, metadata


    def _get_named_entities(self, article):
        # These name indices have the right end point excluded
        names = set()

        if 'context_ner' in article:
            ners = article['context_ner']
            for ner in ners:
                if ner['label'] in ['PERSON', 'ORG', 'GPE']:
                    names.add(ner['text'])

        return names

    def _get_person_names(self, article, pos):
        # These name indices have the right end point excluded
        names = set()

        if 'caption_ner' in article:
            ners = article['caption_ner'][pos]
            for ner in ners:
                if ner['label'] in ['PERSON']:
                    names.add(ner['text'])

        return names

    def _get_caption_names(self, article, idx):
        copy_infos = {}

        parts_of_speech = article['caption_parts_of_speech'][idx]
        caption_ner = article['caption_ner'][idx]
        for pos in parts_of_speech:
            #if pos['pos'] == 'PROPN' and self.is_in_ner(pos['text'], caption_ner):
            if self.is_in_ner(pos['text'], caption_ner):
                if pos['text'] not in copy_infos:
                    copy_infos[pos['text']] = OrderedDict({
                        'caption': [(pos['start'], pos['end'])],
                        'context': []
                    })
                else:
                    copy_infos[pos['text']]['caption'].append(
                        (pos['start'], pos['end']))

        return copy_infos

    def _get_context_names(self, article):
        copy_infos = {}

        context_pos = article['context_parts_of_speech']
        context_ners = article['context_ner']
        for pos in context_pos:
            if pos['pos'] == 'PROPN' and self.is_in_ner(pos['text'], context_ners):
                if pos['text'] not in copy_infos:
                    copy_infos[pos['text']] = OrderedDict({
                        'context': [(pos['start'], pos['end'])]
                    })
                else:
                    copy_infos[pos['text']]['context'].append(
                        (pos['start'], pos['end']))

        return copy_infos

    def _process_copy_tokens(self, copy_infos, article):
        context_pos = article['context_parts_of_speech']
        for name, info in copy_infos.items():
            for pos in context_pos:
                #if pos['pos'] == 'PROPN' and pos['text'] == name:
                if pos['text'] == name:
                    info['context'].append((
                        pos['start'],
                        pos['end'],
                    ))

    def is_in_ner(self, text, ners):
        for ner in ners:
            if text in ner['text']:
                return True
        return False
