import models
import utils
import torch
import os

def prediction(inp):
    """
    Input is a text sequences.Produce scores for the next word in the sequence.
    Scores should be raw logits not post-softmax activations.
    Load your model before generating predictions.
    :param inp: array of words (batch size, sequence length) [0-labels]
    :return: array of scores for the next word (batch size, labels)
    """
    model_path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))),
        'models',
        'drop-out-training.pt')
    #chunk_path = model_path + '.npy.{}'  # format for each chunk
    #data = chunks.read_chunks(chunk_path)
    # Load the data
    #state_dict = chunks.load_from_numpy(data)
    word_count = 33278
    embedding_dim = 200
    hidden_dim = 200
    model = models.LSTMModelSingle(word_count,embedding_dim, hidden_dim)
    #print(model)
    # Load dictionary into your model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(
        utils.to_variable(
            utils.to_tensor(inp).long())).cpu().data.numpy()[
        :, -1, :]
    return out
