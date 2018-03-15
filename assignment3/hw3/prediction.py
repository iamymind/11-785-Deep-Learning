import hw3.models as models
import hw3.chunks as chunks
import hw3.utils as utils
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
        'lr-1e-2-base.pt')
    chunk_path = model_path + '.npy.{}'  # format for each chunk
    data = chunks.read_chunks(chunk_path)
    # Load the data
    state_dict = chunks.load_from_numpy(data)
    model = models.LSTMModelV2(33278, 400, 1150)
    print(model)
    # Load dictionary into your model
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    out = model(
        utils.to_variable(
            utils.to_tensor(inp).long())).cpu().data.numpy()[
        :, -1, :]
    return out
