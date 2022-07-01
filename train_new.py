import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
from six.moves import cPickle
import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import sys
from visual_news import Encoder, Decoder, EncoderLayer, MultiHeadAttentionLayer, PositionwiseFeedforwardLayer, DecoderLayer, Seq2Seq, translate_sentence, bleu, EncoderCNN
try:
    import tensorflow as tf

except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    # summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    # writer.add_summary(summary, iteration)
    if torch.is_tensor(value):
      value = value.cpu()
    with writer.as_default():
      #print(value)
      tf.summary.scalar(key, value, step=iteration)

def train(encoder, model, loader, encoder_optimizer, optimizer, criterion, epoch, logger, logging=True):
    encoder.train()
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()

    start = time.time()

    t = tqdm(train_loader, desc='Train %d' % epoch)

    data = loader.get_batch('train')

    data['images'] = utils.prepro_images(data['images'], True)
    # torch.cuda.synchronize()
    print('Read data:', time.time() - start)

    # torch.cuda.synchronize()
    start = time.time()
    tmp = [data['images'], data['labels'], data['masks']]
 
    for i, (imgs, caps, caplens, image_ids, arts, artlens, mask, reference, lenre, mask2) in enumerate(t):
        #if i ==0:
           #break
        data_time.update(time.time() - start)
        imgs = imgs.to(device)
        caps = caps.to(device)
        arts = arts.to(device)
        reference = reference.to(device)
        imgs = encoder(imgs)

        output, _, mask = model(arts, reference, caps[:, :-1], imgs)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        caps = caps[:, 1:].contiguous().view(-1)
        loss = criterion(output, caps)

        optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        decode_lengths = [c - 1 for c in caplens]
        losses.update(loss.item(), sum(decode_lengths))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        batch_time.update(time.time() - start)

        start = time.time()

    # log into tf series
    print('Epoch [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, losses.avg, np.exp(losses.avg)))
    if logging:
        logger.scalar_summary('loss', losses.avg, epoch)
        logger.scalar_summary('Perplexity', np.exp(losses.avg), epoch)


def main(opt,loader):
    np.random.seed(42)
    warnings.filterwarnings('ignore')

    
    enc = Encoder(loader.vocab_size, 512, 1, 8, 512, 0.1)
    dec = Decoder(loader.vocab_size, 512, 2, 8, 512, 0.1)
    model = Seq2Seq(enc, dec, 0, 0)
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    encoder = EncoderCNN()
    encoder.fine_tune(args.fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.encoder_lr)

    
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    model.apply(initialize_weights)

    model = model.to(device)

    encoder = encoder.to(device)
    if torch.cuda.device_count() > 1 and not isinstance(encoder, torch.nn.DataParallel):
        encoder = torch.nn.DataParallel(encoder)

    criterion = utils.LanguageModelCriterion()

    #criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>']).to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    for epoch in range(args.start_epoch, args.epochs):
        if args.epochs_since_improvement == 20:
            break
        if args.epochs_since_improvement > 0 and args.epochs_since_improvement % 6== 0:
            adjust_learning_rate(optimizer, 0.6)

        train(encoder = encoder,
              model = model,
              train_loader=train_loader,
              criterion=criterion,
              encoder_optimizer = encoder_optimizer,
              optimizer = optimizer,
              epoch=epoch,
              logger=train_logger,
              logging=True)
        if epoch > 4:
            recent_bleu4 = validate(encoder=encoder,
                                    model=model,
                                    val_loader=val_loader,
                                    criterion=criterion,
                                    vocab=vocab,
                                    epoch=epoch,
                                    logger=dev_logger,
                                    logging=True)

            is_best = recent_bleu4 > best_bleu4
            # is_best = 1
            # recent_bleu4 = 1
            # best_bleu4 = 1
            best_bleu4 = max(recent_bleu4, best_bleu4)
            print('learning_rate:', args.decoder_lr)
            if not is_best:
                args.epochs_since_improvement += 1
                print("\nEpoch since last improvement: %d\n" % (args.epochs_since_improvement,))
            else:
                args.epochs_since_improvement = 0

        if epoch <= 4:
            recent_bleu4 = 0
            is_best = 1




    opt.use_att = utils.if_use_att(opt.caption_model)
    #loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    # for debug purposes
    # a=get_batch_one(opt, [loader.split_ix, loader.shuffle, loader.iterators, loader.label_start_ix, loader.label_end_ix])
    # loader.get_batch('train')
    if not os.path.exists(opt.checkpoint_path+'tensorboard/'):
        os.makedirs(opt.checkpoint_path+'tensorboard/')

    else:
        for path in os.listdir(opt.checkpoint_path+'tensorboard/'):
            os.remove(opt.checkpoint_path+'tensorboard/'+path)
    tf_summary_writer = tf and tf.summary.create_file_writer(opt.checkpoint_path+'tensorboard/')
    np.random.seed(42)
    infos = {}
    histories = {}

    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'),'rb') as f:
                histories = cPickle.load(f)

    #sys.exit()
    iteration = infos.get('iter', 0)
    # iteration = 26540
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})

    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    cnn_model = utils.build_cnn(opt)
    cnn_model.cuda()
    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.finetune_cnn_after != -1:
        # only finetune the layer2 to layer4
        cnn_optimizer = optim.Adam([\
            {'params': module.parameters()} for module in cnn_model._modules.values()[5:]\
            ], lr=opt.cnn_learning_rate, weight_decay=opt.cnn_weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        if os.path.isfile(os.path.join(opt.start_from, 'optimizer.pth')):
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            if os.path.isfile(os.path.join(opt.start_from, 'optimizer-cnn.pth')):
                cnn_optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer-cnn.pth')))
    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            # Update the training stage of cnn
            if opt.finetune_cnn_after == -1 or epoch < opt.finetune_cnn_after:
                for p in cnn_model.parameters():
                    p.requires_grad = False
                cnn_model.eval()
            else:
                for p in cnn_model.parameters():
                    p.requires_grad = True
                # Fix the first few layers:
                for module in cnn_model._modules.values()[:5]:
                    for p in module.parameters():
                        p.requires_grad = False
                cnn_model.train()
            update_lr_flag = False
        # torch.cuda.synchronize()
        start = time.time()
        # Load data from train split (0)
        # for validation training change the split to 'val'
        # data = loader.get_batch('val')
        data = loader.get_batch('train')

        data['images'] = utils.prepro_images(data['images'], True)
        # torch.cuda.synchronize()
        print('Read data:', time.time() - start)

        # torch.cuda.synchronize()
        start = time.time()
        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = cnn_model(images).permute(0, 2, 3, 1) # change order of output
        fc_feats = att_feats.mean(2).mean(1)


        if not opt.use_att:
            att_feats = Variable(torch.FloatTensor(1, 1,1,1).cuda())

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), opt.seq_per_img,) +
                                                    att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * opt.seq_per_img,)
                                                                                               + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), opt.seq_per_img,) +
                                                  fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * opt.seq_per_img,) +
                                                                                            fc_feats.size()[1:]))
        model.zero_grad()
        optimizer.zero_grad()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            cnn_optimizer.zero_grad()

        if opt.sentence_embed:
            sen_embed = Variable(torch.from_numpy(np.array(data['sen_embed'])).cuda())
            out = model(fc_feats, att_feats, labels, sen_embed)
            loss = crit(out, labels[:, 1:], masks[:, 1:])
            # loss += cov
        else:
            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
               # - 0.001 * crit(model(torch.zeros(fc_feats.size()).cuda(), torch.zeros(att_feats.size()).cuda(), labels), labels[:,1:], masks[:,1:])
        loss.backward()
        # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
            utils.clip_gradient(cnn_optimizer, opt.grad_clip)
            cnn_optimizer.step()
        # train_loss = loss.data[0]
        train_loss = loss.item()
        # torch.cuda.synchronize()

        end = time.time()
        print("Step [{}/{}], Epoch [{}/{}],  train_loss = {:.3f}, time/batch = {:.3f}" \
            .format((iteration+1)%int(len(loader)/vars(opt)['batch_size']), int(len(loader)/vars(opt)['batch_size']),
                     epoch, vars(opt)['max_epochs'], train_loss, end - start))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opt.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k,v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush()
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model.pth')
                cnn_checkpoint_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'model-cnn.pth')
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                print("cnn model saved to {}".format(cnn_checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)
                if opt.finetune_cnn_after != -1 and epoch >= opt.finetune_cnn_after:
                    cnn_optimizer_path = os.path.join(opt.checkpoint_path + opt.caption_model, 'optimizer-cnn.pth')
                    torch.save(cnn_optimizer.state_dict(), cnn_optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path + opt.caption_model, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path + opt.caption_model, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path+ opt.caption_model, 'model-best.pth')
                    cnn_checkpoint_path = os.path.join(opt.checkpoint_path+ opt.caption_model, 'model-cnn-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    torch.save(cnn_model.state_dict(), cnn_checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break
        #sys.exit()


opt = opts.parse_opt()
loader = DataLoader(opt)
train(opt,loader)

