
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
from six.moves import cPickle
import sys
try:
    import tensorflow as tf

except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None
import pickle as pkl
from encoder import Encoder
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
#from adaptive_loss import AdaptiveLoss

# In[16]:



import argparse
import torchvision
import re

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    parser.add_argument('--input_json', type=str, default='/home/jupyter/google/data/data_news.json', #data/cocotalk.json, data/data_europeana.json
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_label_h5', type=str, default='/home/jupyter/google/data/data_news_label.h5', # data_europeana_label.h5
                    help='path to the h5file containing the preprocessed label')
    parser.add_argument('--input_image_h5', type=str, default='/home/jupyter/google/data/data_news_image.h5', # data_europeana_image.h5
                    help='path to the h5file containing the preprocessed image')
    parser.add_argument('--cnn_model', type=str, default='resnet152',
                    help='resnet')
    parser.add_argument('--cnn_weight', type=str, default='/home/jupyter/google/data/resnet152-394f9c45.pth',
                    help='path to CNN tf model. Note this MUST be a resnet right now.')
    parser.add_argument('--start_from', type=str, default=None,
                        #default='./save/show_attend_tell',
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)

    # Model settings
    parser.add_argument('--caption_model', type=str, default="show_attend_tell",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')
    parser.add_argument('--rnn_size', type=int, default=512,
                    help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in the RNN')
    parser.add_argument('--rnn_type', type=str, default='lstm',
                    help='rnn, gru, or lstm')
    parser.add_argument('--input_encoding_size', type=int, default=512,
                    help='the encoding size of each token in the vocabulary, and the image.')
    parser.add_argument('--att_hid_size', type=int, default=512,
                    help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type=int, default=2048,
                    help='2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type=int, default=2048,
                    help='2048 for resnet, 512 for vgg')
    parser.add_argument('--sentence_embed', type=str, default='/home/jupyter/google/data/articles_full_USE.h5', #/media/abiten/SSD-DATA/breakingnews
                        help='it can be either LDA or SkipThought')
    parser.add_argument('--sentence_embed_att', type=bool, default=True,
                        help='Use attention or not')
    parser.add_argument('--sentence_embed_method', type=str, default='fc',
                        help='choose which method to use, available options are fc_max, conv, conv_deep, fc, bnews')
    parser.add_argument('--sentence_length', type=int, default=54,
                        help='hyperparameter to pad the values, this is used for both the sentence and word level')
    parser.add_argument('--sentence_embed_size', type=int, default=512,
                        help='size for sentence embedding')

    # Optimization: General
    parser.add_argument('--max_epochs', type=int, default=150,
                    help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                    help='minibatch size')
    parser.add_argument('--grad_clip', type=float, default=5.0, #5.,
                    help='clip gradients at this value')
    parser.add_argument('--num_thread', type=int, default=4,
                        help='Number of threads to be used for retrieving the data')
    parser.add_argument('--drop_prob_lm', type=float, default=0.2,
                    help='strength of dropout in the Language Model RNN')
    parser.add_argument('--finetune_cnn_after', type=int, default=-1,
                    help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
    parser.add_argument('--seq_per_img', type=int, default=1,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
    parser.add_argument('--beam_size', type=int, default=1,
                    help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=30,
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=8,
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.8,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')

    #Optimization: for the CNN
    parser.add_argument('--cnn_optim', type=str, default='adam',
                    help='optimization to use for CNN')
    parser.add_argument('--cnn_optim_alpha', type=float, default=0.8,
                    help='alpha for momentum of CNN')
    parser.add_argument('--cnn_optim_beta', type=float, default=0.999,
                    help='beta for momentum of CNN')
    parser.add_argument('--cnn_learning_rate', type=float, default=1e-5,
                    help='learning rate for the CNN')
    parser.add_argument('--cnn_weight_decay', type=float, default=0,
                    help='L2 weight decay just for the CNN')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=5000,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1000,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save/',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=100,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.save_checkpoint_every > 0, "save_checkpoint_every should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        #self.criterion = nn.KLDivLoss(reduction="sum")
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        #print(x.shape, target.shape)
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

def evaluate_bleu(gen_texts_2, captions_2, eval_out):
    for gen, ref in zip(gen_texts_2, captions_2):
        bleu_scorer = BleuScorer(n=4)
        bleu_scorer += (gen, [ref])
        score, _ = bleu_scorer.compute_score(option='closest')

        eval_out['bleu-1'] += score[0] * 100
        eval_out['bleu-2'] += score[1] * 100
        eval_out['bleu-3'] += score[2] * 100
        eval_out['bleu-4'] += score[3] * 100
        #score = evaluate({1:[ref]},{1:[gen]} )
        #eval_out['bleu-1'] += score['Bleu_1'] * 100
        #eval_out['bleu-2'] += score['Bleu_2'] * 100
        #eval_out['bleu-3'] += score['Bleu_3'] * 100
        #eval_out['bleu-4'] += score['Bleu_4'] * 100

        #cider_scorer = CiderScorer(n=4, sigma=6.0)

        #score, _ = Cider().compute_score({1:[ref]},{1:[gen]})
        #eval_out['CIDEr'] += score * 100
        #print(score)
    return eval_out

#from final.transformer import Batch
def run_epoch(loader, encoder, model, loss_compute, epoch, scheduler, optimizer, dataset, mode='train', size_per_epoch=5000):

    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    iteration = 0
    predictions = []
    print(len(loader))
    n_samples = 0
    eval_out = {'bleu-1':0.0,'bleu-2':0.0,'bleu-3':0.0,'bleu-4':0.0,'bleu-1r':0.0, 'CIDEr':0.0}
    start = 0
    if mode == 'train':
        max_epochs = (len(loader)//size_per_epoch) + 1
        start = size_per_epoch*(epoch%(max_epochs))
    else:
        cider_scorer = CiderScorer(n=4, sigma=6.0)
        
    #print(start)
    for target_batch, target_mask, label_batch, image_batch, context_batch, ntokens, metadata in loader:
        n_samples += target_batch.shape[0]
        start_batch = time.time()
        #data = loader.get_batch(mode)
        #data['images'] = utils.prepro_images(data['images'], True)
        #tmp = [data['images'], data['labels'], data['masks']]
        #tmp = [Variable(torch.from_numpy(_), requires_grad=False).to(device) for _ in tmp]
        #images, labels, masks = tmp

        #images = images.to(device)
        ##print(labels.shape)
        #images_feats = encoder(images)

        #batch = final.transformer_new.Batch(trg=labels)
        #sen_embed = Variable(torch.from_numpy(np.array(data['sen_embed'])).cuda())
        #print(image_batch)
        memory = encoder.forward(image_batch, context_batch, metadata)
        #print('Encoder time:', time.time()-start_batch)
        #print(target_batch)
        target_batch = target_batch.to(device)
        target_mask = target_mask.to(device)
        label_batch = label_batch.to(device)
        out = model.forward(memory, target_batch, 
                            None, target_mask)
        #torch.cuda.synchronize()
        #print('Decoder time:', time.time()-start_batch)


        #print(out.shape)
        loss, loss_node = loss_compute(out, label_batch, ntokens)
        #torch.cuda.synchronize()
        #print('Loss time:', time.time()-start_batch)

        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            optimizer.zero_grad(set_to_none=True)
            loss_node.backward()
            #print('Backprop time:', time.time()-start_batch)
            #train_state.step += 1
            #train_state.samples += batch.src.shape[0]
            #train_state.tokens += batch.ntokens
            optimizer.step()
            n_accum += 1
            #train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if iteration % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Iter Step: %d/%d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e | Time/Batch: %7.2f"
                )
                % (epoch, (iteration+1), int(len(loader)), 
                   loss / ntokens, tokens / elapsed, lr, time.time()-start_batch)
            )
            start = time.time()
            tokens = 0
        if iteration % 40 == 1 and (mode == "val"):
            elapsed = time.time() - start
            print(
                (
                    "Validation Epoch Step: %6d | Iter Step: %d/%d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Time/Batch: %7.2f | nsamples:%d"
                )
                % (epoch, (iteration+1), int(len(loader)), 
                   loss / ntokens, tokens / elapsed, time.time()-start_batch, n_samples)
            )
            start = time.time()
            tokens = 0
        iteration += 1
        if mode == 'val':
          outputs = translate_sentence(model, memory, dataset.tokenizer, 50)#model, image_feats, word_map, max_len
          #outputs = translate_sentence(model, memory, dataset.vocab['word2idx'], dataset.vocab['idxtoword'], 50)#model, image_feats, word_map, max_len
          captions = [m['caption'] for m in metadata]
          gen_texts_2 = [re.sub(r'[^\w\s]', '', t) for t in outputs]
          captions_2 = [re.sub(r'[^\w\s]', '', t) for t in captions]
          evaluate_bleu(gen_texts_2, captions_2, eval_out)
          for gen, ref in zip(gen_texts_2, captions_2):
            cider_scorer += (gen, [ref])
          if iteration % 40 == 0:
              print(gen_texts_2[0])
              print(captions_2[0])

          #for k,sent in enumerate(outputs):
          #  entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'image_path'
          #          : data['infos'][k]['file_path']}
          #  predictions.append(entry)
          #if iteration == 7000:
          #  break
          

        del loss
        del loss_node
        torch.cuda.empty_cache()
        if iteration > size_per_epoch:
            break
        if mode == 'val' and n_samples > 5000:
            break
        #if epoch == 0 and mode == 'train':
        #    break
        #if data['bounds']['wrapped']:
        #  break
        #break
    if mode == 'val':
        print(gen_texts_2[:3])
        for key, value in eval_out.items():
            eval_out[key] = value / n_samples
        cider_score, _ = cider_scorer.compute_score()
        eval_out['CIDEr'] = cider_score * 100
        print(eval_out)
    return total_loss / total_tokens, eval_out

import transformer_v2
from torch.optim.lr_scheduler import LambdaLR
import GPUtil
from dataloader import collate_fn
from dataloader import GoodNewsDataset
from torch.utils.data import DataLoader

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None
    def step(self):
        None
    def zero_grad(self, set_to_none=False):
        None

class DummyScheduler:
    def step(self):
        None

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator.proj(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def train(opt):
    dataset = GoodNewsDataset()
    val_dataset= GoodNewsDataset(split='val')
    loader = DataLoader(dataset, batch_size=vars(opt)['batch_size'], collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=vars(opt)['batch_size'], collate_fn=collate_fn)
    vocab_size = dataset.tokenizer.vocab_size
    pad_idx = 0
    d_model = 512
    model = transformer_v2.make_model_news(vocab_size+1, num_enc_dec=3, dim_model = d_model, dim_feedfwd=2048, img_dim=1024, sent_dim=1024)
    
    model = model.cuda()
    encoder = Encoder()
    model_path = 'v3/latest.pt'
    sch_path =  'v3/scheduler.pth.tar'
    perf_pkl = 'v3/best_model.pkl'
    info_pkl = 'v3/model_info.pkl'
    best_model_path = 'v3/best.pt'
    criterion = LabelSmoothing(vocab_size+1, padding_idx=pad_idx, smoothing=0.0)
    #criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="sum")
    criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=30000
        ),
    )
    epoch_cont = 0
    if os.path.exists(model_path):
      model.load_state_dict(torch.load(model_path, map_location=device))
      lr_scheduler.load_state_dict(torch.load(sch_path))
      with open(info_pkl,'rb') as bfile:
          info = pkl.load(bfile)
          epoch_cont = info['epoch']+1
    best_bleu4 = 0
    if os.path.exists(perf_pkl):
        with open(perf_pkl,'rb') as bfile:
            best_bleu4 = pkl.load(bfile)['bleu-4']
    
    for i in range(epoch_cont, 80):
      print(f"Epoch {i} Train ====", flush=True)
      model.train()
      tloss,_temp = run_epoch(loader,encoder,model,SimpleLossCompute(model.generator, criterion), i, lr_scheduler, optimizer, dataset)
      print(f"Epoch {i} Train ==== Avg loss: {tloss}", flush=True)
      torch.cuda.empty_cache()
      GPUtil.showUtilization()
      print(f"Epoch {i} Validation ====", flush=True)
      model.eval()
      sloss,eval_out = run_epoch(
            val_loader,encoder,
            model,
            SimpleLossCompute(model.generator, criterion),i,DummyScheduler(),
            DummyOptimizer(), val_dataset,
            mode="val"
        )
      bleu4 = eval_out['bleu-4']
      if bleu4 > best_bleu4:
        best_bleu4 = bleu4
        torch.save(model.state_dict(), best_model_path)
        with open(perf_pkl,'wb') as bfile:
            pkl.dump(eval_out, bfile)
      print(sloss)

      torch.save(model.state_dict(), model_path)
      torch.save(lr_scheduler.state_dict(), sch_path)
      with open(info_pkl,'wb') as bfile:
          info = {'epoch':i}
          pkl.dump(info, bfile)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice

def evaluate(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
        # (Spice(), "Spice")
    ]
    final_scores = {}
    #print(len(ref[1][0]), len(hypo[1][0]))
    #print(ref)
    #print(ref)
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    return final_scores

def language_eval(preds, split):                                   
    import sys                                                                        
                                                                          
    # TODO: NYTIMES                                                               
    if split == 'val':                                                            
        annFile = '/home/jupyter/google/data/val.json'                                               
        with open(annFile, 'rb') as f: dataset = json.load(f)                     
    else:                                                                         
        annFile = '/home/jupyter/google/data/test.json'                                              
        with open(annFile, 'rb') as f: dataset = json.load(f)                     
                                                                                  
    # TODO: BREAKINGNEWS                                                          
    # with open("/home/abiten/Desktop/Thesis/newspaper/breakingnews/bnews_caps.json", "rb") as f: dataset = json.load(f)
    #print(len(preds))
    id_to_ix = {v['cocoid']: ix for ix, v in enumerate(dataset)}                  
    hypo = {v['image_id']: [v['caption']] for v in preds if v['image_id'] in id_to_ix}
    ref = {k: [i['raw'] for i in dataset[id_to_ix[k]]['sentences']] for k in hypo.keys() if k in id_to_ix}
    final_scores = evaluate(ref, hypo)                                            
    print('Bleu_1:\t', final_scores['Bleu_1'])
    print('Bleu_2:\t', final_scores['Bleu_2'])                                    
    print('Bleu_3:\t', final_scores['Bleu_3'])                                    
    print('Bleu_4:\t', final_scores['Bleu_4'])                                    
    # print('METEOR:\t', final_scores['METEOR'])                                  
    print('ROUGE_L:', final_scores['ROUGE_L'])                                    
    print('CIDEr:\t', final_scores['CIDEr'])
    # print('Spice:\t', final_scores['Spice'])
    return final_scores


def decode_sequence(tokenizer, seq, start=0):                                                 
    N, D = seq.size()
    #print(seq.size())
    out = []                                                                          
    for i in range(N):
        
        tokens = []
        for j in range(start,D):
            ix = seq[i,j].cpu().numpy()                                               
            if ix == int(tokenizer.sep_token_id):
                break                                                                 
            tokens.append(int(ix))
        txt = tokenizer.decode(tokens, skip_special_tokens=True)
        #for j in range(start,D):
        #    ix = seq[i,j].cpu().numpy()                                               
        #    if ix == int(tokenizer.sep_token_id):
        #        break                                                                 
        #    #if ix > 0 :
        #    #if ix == 0:
        #    #  txt = txt + '<pad>'
        #    #else:
        #    if ix >= 2:
        #      txt = txt + ' ' + ix2word[int(ix)]
        #      #txt = txt + 
        out.append(txt)
    #for i in range(3):
    #    print(out[i])
    return out


def translate_sentence(model, image_feats, tokenizer, max_len):         
    model.eval()                                
    ys = torch.zeros(opt.batch_size, 1).fill_(int(tokenizer.cls_token_id)).type(torch.LongTensor).to(device)
    for i in range(max_len - 1):
        out = model.decode(
            image_feats, None, ys, transformer_v2.subsequent_mask(ys.size(1)).to(device)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        #next_word = next_word.data[0]
        ys = torch.cat(
            [ys, next_word.unsqueeze(1)], dim=1
        )
        #print(ys.shape)
    outputs = decode_sequence(tokenizer, ys, start=1)
    return outputs


    outputs = [int(word2idx['<start>'])] * opt.batch_size
    #imgs = model.relu(model.l1(imgs))
    batch = final.transformer_new.Batch(trg=labels)

    out = model.forward(images_feats, batch.trg, 
                        None, batch.trg_mask)                                        
    # reference = enc_ref
    # with torch.no_grad():
    #     enc_src = model.encoder(src, src_mask, imgs)
    #     enc_ref = model.encoder(enc_ref, ref_mask, imgs)                              
    
    max_length = max(caplens)
    
    #outputs = [1]                                           
    for i in range(max_length):

        # torch.Size([32, 1])
        # torch.Size([32])
        # torch.Size([32, 2])
        # torch.Size([32, 1, 2, 2])  
        
        if i==0:
          trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
        else:
          #print(trg_tensor.shape)
          #print(best_guess.shape)
          trg_tensor = torch.cat((trg_tensor,best_guess.unsqueeze(1)),dim=1).to(device)
        #print(trg_tensor.shape)
        trg_mask = model.make_trg_mask(trg_tensor)
        #print(trg_mask.shape)

        with torch.no_grad(): 
            output, attention = model.decoder(trg_tensor, src, trg_mask, src_mask, imgs)

        #TODO handle eos in batch                                                                              
        best_guess = output.argmax(2)[:, -1]
        #print(best_guess.shape)                                   
        #outputs.append(best_guess)                                                    
                                                                                      
        #if best_guess == int(word_map.word2idx['<end>']):                                  
        #    break
    trg_tensor = torch.cat((trg_tensor,best_guess.unsqueeze(1)),dim=1).to(device)                                                                     
    outputs = decode_sequence(word2idx, trg_tensor, start=1)
    # print(trg_tensor.shape)
    # print(outputs)
    # print(x,y,z)
    #translated_sentence = [word_map.ix_to_word[str(idx)] for idx in outputs]                 
    # remove start token                                                              
    #return translated_sentence[1:]   
    return outputs


opt = parse_opt()
#import dataloader
torch.cuda.empty_cache()
#loader = dataloader.make_loader(batch_size=vars(opt)['batch_size'])
train(opt)


