import torch
import numpy as np
from torch.autograd import Variable

def save_model(model, path):
    torch.save(model.state_dict(), path)

def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).long()

def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)

def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))
def batchify(data, bsz):
    # Taken from pytorch example
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def custom_data_loader(data, args, evaluation=False):
    #adopted from pytorch example
    index = 0
    seq_len = 0 
    while (index < len(data)-1):
        if evaluation:
            seq_len = min(args.base_seq_len, len(data) - 1 - index)
        else:
            # sample a bernoulli
            m = np.random.binomial(1,args.seq_prob, 1)
            seq_len = args.base_seq_len if m else args.base_seq_len//2
            seq_len = max(args.min_seq_len, int(np.random.normal(seq_len, args.seq_std)))
            seq_len = min(seq_len, len(data) - 1 - index)
        X    = Variable(data[index:index+seq_len], volatile=evaluation)
        y    = Variable(data[index+1:index+1+seq_len])
        index += seq_len

        yield (X, y, seq_len)
def to_text(preds, vocabulary):
    return ["".join(vocabulary[c] for c in line) for line in preds]
    
def print_generated(lines):
    for i, line in enumerate(lines):
        print("Generated text {}: {}".format(i, line))
            
def generate(model, sequence_length, batch_size, args, stochastic=False, inp=None):
    if inp is None:
        inp = Variable(torch.zeros(batch_size, 1)).long()
        if args.cuda:
            inp = inp.cuda()
    model.eval()
    logits = model(inp, forward=sequence_length, stochastic=stochastic)
    classes = torch.max(logits, dim=2)[1]
    return classes