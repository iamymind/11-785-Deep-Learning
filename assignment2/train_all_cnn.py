"""
Refer to handout for details.
- Build scripts to train your model
- Submit your code to Autolab
"""
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F

import numpy as np
import hw2.all_cnn
import hw2.preprocessing

def write_results(predictions, output_file='predictions.txt'):
    """
    Write predictions to file for submission.
    File should be:
        named 'predictions.txt'
        in the root of your tar file
    :param predictions: iterable of integers
    :param output_file:  path to output file.
    :return: None
    """
    with open(output_file, 'w') as f:
        for y in predictions:
            f.write("{}\n".format(y))

def load_data(path):

    x      = np.load(path + 'train_feats.npy')
    labels = np.load(path + 'train_labels.npy')
    xtest  = np.load(path + 'test_feats.npy')
    #N = 100
    x, xtest = hw2.preprocessing.cifar_10_preprocess(x, xtest, image_size=32)

    return x,labels,xtest	

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()
    		
def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)
        		
def save_model(model, path):
    torch.save(model.state_dict(), path)

def main(tag):

    lr = 1e-2
    L2 = 1e-3
    n_epochs = 50
    momentum = 0.9
    batch_size = 128 
    data_path = './dataset/'
    log_path  = './logs/' + tag
    model_path = './models/'+ tag + '.pt'

    model = hw2.all_cnn.all_cnn_module()        
    optimizer = torch.optim.SGD(model.parameters(), 
    lr=lr,momentum=momentum,nesterov=True,weight_decay=L2)
    loss_fn = nn.NLLLoss() #

    if torch.cuda.is_available():
        model = model.cuda()   
        loss_fn = loss_fn.cuda()

    train_feats, train_labels , test_feats = load_data(data_path)

    train_size = train_labels.shape[0]
    print('Data loading done')
    train = data_utils.TensorDataset(to_tensor(train_feats),
             to_tensor(train_labels))
    train_loader = data_utils.DataLoader(train, 
    batch_size=batch_size, shuffle=True)

    log_numpy = []
            
    print('Done testing !!')
    
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        correct = 0
        model.train()
        for batch_index, (data, label) in enumerate(train_loader):

            optimizer.zero_grad()

            X, Y = to_variable(data), to_variable(label)

            out  = F.log_softmax(model(X.view(label.size()[0], 3, 32, 32)))
            pred = out.data.max(1, keepdim=True)[1].int()
            predicted = pred.eq(Y.data.view_as(pred).int())

            correct += predicted.sum()
            loss = loss_fn(out, Y.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data.sum()
            if (batch_index % 100)==0: 
                print("batchs left : ",int(train_size/batch_size-batch_index))

        total_loss  = epoch_loss*batch_size/train_size
        train_error = 1 - correct/train_size
        log_numpy.append([total_loss,train_error])

        print("epoch: {:f}, loss: {:f}, error: {:f}".format(epoch+1, total_loss, train_error))

        try:
            save_model(model,model_path)
        except:
            print("dumping model error!!")

        try:
            np.save(log_path+'.npy',np.array(log_numpy))
        except:
            print("dumping log error!!")

    try:
        model.eval()
        test_feats = to_variable(to_tensor(test_feats))
        out  = F.log_softmax(model(test_feats))
        pred = out.data.max(1, keepdim=True)[1].int()
        write_results(pred.numpy().tolist(), tag)
    except:
        print("Testing failed !!")


if __name__ == '__main__':
	main('train1_5epochs_lr_2')
