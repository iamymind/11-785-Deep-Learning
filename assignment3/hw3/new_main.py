import argparse
import os
import sys
import math
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils
import models
from torch.autograd import Variable
import torch
from collections import Counter

def validate(model, val_loader, loss_fn, n_batchs, word_count):

    model.eval()
    val_loss = 0
    batch_index = 0
    counter = 0
    #hidden = model.init_hidden(20)
    while (batch_index < n_batchs - 1):
        X, y, seq_len = next(val_loader)
        out = model(X)
        loss = loss_fn(out.view(-1, word_count), y)
        batch_index += seq_len
        val_loss += loss.data.sum()
        counter+=1
        #hidden = model.init_hidden(20)
    return val_loss/counter

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def main(argv):
    parser = argparse.ArgumentParser(
        description='WikiText-2 language modeling')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 90)'),
    parser.add_argument('--eval-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 50)'),
    parser.add_argument(
        '--save-directory',
        type=str,
        default='output/wikitext-2',
        help='output directory')
    parser.add_argument('--model-save-directory', type=str, default='models/',
                        help='output directory')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--base-seq-len', type=int, default=35, metavar='N',
                        help='Batch length'),
    parser.add_argument('--min-seq-len', type=int, default=35, metavar='N',
                        help='minimum batch length'),
    parser.add_argument('--seq-prob', type=int, default=0.95, metavar='N',
                        help='prob of being divided by 2'),
    parser.add_argument('--seq-std', type=int, default=5, metavar='N',
                        help='squence length std'),
    parser.add_argument('--hidden-dim', type=int, default=400, metavar='N',
                        help='Hidden dim')
    parser.add_argument('--embedding-dim', type=int, default=400, metavar='N',
                        help='Embedding dim')
    parser.add_argument('--lr', type=int, default=20, metavar='N',
                        help='learning rate'),
    parser.add_argument('--weight-decay', type=int, default=2e-6, metavar='N',
                        help='learning rate'),
    parser.add_argument(
        '--tag',
        type=str,
        default='hopefully_best_new.pt',
        metavar='N',
        help='learning rate'),
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # load dataset
    train_data, val_data, vocabulary = (
        np.load('./dataset/wiki.train.npy'),
        np.load('./dataset/wiki.valid.npy'),
        np.load('./dataset/vocab.npy')
    )

    word_count = len(vocabulary)

    #model = models.RNNModel(word_count, args)
    loss_fn = torch.nn.CrossEntropyLoss()

    checkpoint_path = os.path.join(args.model_save_directory, args.tag)

    if not os.path.exists(checkpoint_path):
        model = models.LSTMModelSingle(word_count, args.embedding_dim, args.hidden_dim)
    else:
        print("Using pre-trained model")
        print("*" * 90)
        model = models.LSTMModelSingle(word_count, args.embedding_dim, args.hidden_dim)
        checkpoint_path = os.path.join(args.model_save_directory, args.tag)
        model.load_state_dict(torch.load(checkpoint_path))

    if args.cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    '''
    generated = utils.generate(
        model,
        sequence_length=10,
        batch_size=2,
        stochastic=True,
        args=args).data.cpu().numpy()
    utils.print_generated(
        utils.to_text(
            preds=generated,
            vocabulary=vocabulary))
    '''
    print('Model: ', model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    logging = dict()
    logging['loss'] = []
    logging['train_acc'] = []
    logging['val_loss'] = []

    model.train()
    #hidden = model.init_hidden(args.batch_size)
    

    for epoch in range(args.epochs):

        epoch_time = time.time()
        np.random.shuffle(train_data)
        train_data_ = utils.batchify(
            utils.to_tensor(
                np.concatenate(train_data)),
            args.batch_size)
        val_data_ = utils.batchify(
            utils.to_tensor(
                np.concatenate(val_data)),
            args.eval_batch_size)
        train_data_loader = utils.custom_data_loader(train_data_, args, evaluation=True)

        val_data_loader = utils.custom_data_loader(val_data_, args, evaluation=True)
        # number of words
        train_size = train_data_.size(0) * train_data_.size(1)
        val_size = val_data_.size(0) * val_data_.size(1)

        n_batchs = len(train_data_)
        n_batchs_val = len(val_data_)
        correct = 0
        epoch_loss = 0
        batch_index = 0
        seq_len = 0
        counter = 0
        #hidden = model.init_hidden(args.batch_size)
        while (batch_index < n_batchs - 1):

            #optimizer.zero_grad()

            X, y, seq_len = next(train_data_loader)
            #print('X: ', X.shape, 'y: ', y.shape)
            #hidden = repackage_hidden(hidden)
            #out, hidden = model(X, hidden)
            model.zero_grad()

            #out, hidden = model(X, hidden)
            out = model(X)
            
            loss = loss_fn(out.view(-1, word_count), y)
            
            loss.backward()
            # scale lr with respect the size of the seq_len
            #utils.adjust_learning_rate(optimizer, args, seq_len)
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)
                
            #optimizer.step()
            #utils.adjust_learning_rate(optimizer, args, args.base_seq_len)

            epoch_loss += loss.data.sum()
            batch_index += seq_len
            if counter%200==0 and counter!=0:
                print('|batch {:3d}|train loss {:5.2f}|'.format(
                        counter, 
                        epoch_loss/counter))
                
            counter += 1

        train_loss = epoch_loss / counter
        val_loss = validate(model, val_data_loader, loss_fn, n_batchs_val, word_count) 

        logging['loss'].append(train_loss)
        logging['val_loss'].append(val_loss)
        utils.save_model(model, checkpoint_path)

        print('=' * 83)
        print(
            '|epoch {:3d}|time: {:5.2f}s|valid loss {:5.2f}|'
            'train loss {:8.2f}'.format(
                epoch + 1,
                (time.time() - epoch_time),
                val_loss,
                train_loss))


if __name__ == '__main__':
    main(sys.argv[1:])
