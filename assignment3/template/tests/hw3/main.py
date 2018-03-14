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

def validate(model, val_loader, loss_fn, n_batchs, eval_batch_size=50):
    
    model.eval() 
    batch_index =0 
    hidden = model.init_hidden(eval_batch_size)
    val_loss = 0
    counter = 0 
    while (batch_index < n_batchs-1):
        
        X, y, seq_len = next(val_loader)
        out, hidden = model(X, hidden)
        val_loss += loss_fn(out, y)
        hidden = utils.repackage_hidden(hidden)
        batch_index+= seq_len
        counter+=1
        
    return val_loss/counter

#def train(args, train_data, val_data, model, optimizer, loss_fn):

def main(argv):
    parser = argparse.ArgumentParser(description='WikiText-2 language modeling')
    parser.add_argument('--batch-size', type=int, default=70, metavar='N',
                        help='input batch size for training (default: 90)'),
    parser.add_argument('--eval-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for training (default: 50)'),
    parser.add_argument('--save-directory', type=str, default='output/wikitext-2',
                        help='output directory')
    parser.add_argument('--model-save-directory', type=str, default='./models/',
                        help='output directory')    
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--base-seq-len', type=int, default=70, metavar='N',
                        help='Batch length'),
    parser.add_argument('--min-seq-len', type=int, default=50, metavar='N',
                        help='minimum batch length'),
    parser.add_argument('--seq-prob', type=int, default=0.95, metavar='N',
                        help='prob of being divided by 2'),
    parser.add_argument('--seq-std', type=int, default=6, metavar='N',
                        help='squence length std'),
    parser.add_argument('--hidden-dim', type=int, default=1150, metavar='N',
                        help='Hidden dim')
    parser.add_argument('--embedding-dim', type=int, default=400, metavar='N',
                        help='Embedding dim')
    parser.add_argument('--lr', type=int, default=1e-4, metavar='N',
                        help='learning rate'),
    parser.add_argument('--weight-decay', type=int, default=2e-6, metavar='N',
                        help='learning rate'),
    parser.add_argument('--tag', type=str, default='testing.pt', metavar='N',
                        help='learning rate'),
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    #load dataset
    train_data, val_data, vocabulary = (
                                        np.load('./dataset/wiki.train.npy')[:2],
                                        np.load('./dataset/wiki.valid.npy')[:1],
                                        np.load('./dataset/vocab.npy')
                                       )
    
    word_count = len(vocabulary)
    
    model     = models.LSTMModelV2(word_count, args.embedding_dim, args.hidden_dim)
    loss_fn   = models.CrossEntropyLoss3D()
    
    checkpoint_path = os.path.join(args.model_save_directory, args.tag)

    if not os.path.exists(checkpoint_path):
        model = models.LSTMModelV2(word_count, args.embedding_dim, args.hidden_dim)
    else:
        print("Using pre-trained model")
        print("*"*90)
        model = models.LSTMModelV2(word_count, args.embedding_dim, args.hidden_dim)
        checkpoint_path = os.path.join(args.model_save_directory, args.tag)
        model.load_state_dict(torch.load(checkpoint_path))
    
    if args.cuda:
            model = model.cuda()
            loss_fn = loss_fn.cuda()
            
    #generated = utils.generate(model, sequence_length=10, batch_size=2, stochastic=True, args=args).data.cpu().numpy()
    #utils.print_generated(utils.to_text(preds=generated, vocabulary=vocabulary))
    print('Model: ', model)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    logging = dict()
    logging['loss']=[]
    logging['train_acc']=[]
    logging['val_acc']=[]
    
    
    model.train()
    
    for epoch in range (args.epochs):
        
        epoch_time = time.time()    
        np.random.shuffle(train_data)
        train_data_   = utils.batchify(utils.to_tensor(np.concatenate(train_data)), args.batch_size)
        val_data_     = utils.batchify(utils.to_tensor(np.concatenate(val_data)), args.eval_batch_size)
        train_data_loader  = utils.custom_data_loader(train_data_, args)
        val_data_loader    = utils.custom_data_loader(val_data_, args, evaluation=True)
        #number of words
        train_size = train_data_.size(0)*train_data_.size(1)
        val_size   = val_data_.size(0)*val_data_.size(1)

        n_batchs      = len(train_data_)
        n_batchs_val  = len(val_data_)
        correct = 0
        epoch_loss = 0
        batch_index = 0
        seq_len = 0
        counter = 0 
        hidden = model.init_hidden(args.batch_size)

        while (batch_index < n_batchs-1):

            optimizer.zero_grad() 

            X, y, seq_len = next(train_data_loader)
            
            hidden = utils.repackage_hidden(hidden)
            model.zero_grad()
            out, hidden = model(X, hidden)
            loss = loss_fn(out, y)

            pred      = out.data.max(2)[1].int().view(1,-1)
            predicted = pred.eq(y.data.view_as(pred).int())
            correct  += predicted.sum()

            loss.backward()
            #scale lr with respect the size of the seq_len
            utils.adjust_learning_rate(optimizer, args, seq_len)
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            for p in model.parameters():
                p.data.add_(-args.lr, p.grad.data)
            optimizer.step()
            utils.adjust_learning_rate(optimizer, args, args.base_seq_len)
            
            epoch_loss  += loss.data.sum()
            batch_index += seq_len
            counter +=1
            break
        train_acc   = correct/train_size      
        train_loss  = epoch_loss/counter
        val_loss     = validate(model, val_data_loader, loss_fn, n_batchs_val)
        
        logging['loss'].append(train_loss)
        logging['train_acc'].append(train_acc)
        logging['val_acc'].append(val_loss)
        utils.save_model(model, checkpoint_path)

            
        print('=' * 83)
        #print('| epoch {:3d} | time: {:5.2f}s | valid acc {:5.2f} | 
        #      train acc {:5.2f} |train loss {:8.2f}'.format(epoch+1, (time.time() - epoch_time),
        #                                   val_loss, train_acc, train_loss))
        print('val_loss: ', val_loss)
        break
            
if __name__ == '__main__':
    main(sys.argv[1:])
