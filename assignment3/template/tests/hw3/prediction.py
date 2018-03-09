import hw3.models as models 
import hw3.chunks as chunks 
import hw3.utils  as utils 

def prediction(inp):
    """
    Input is a text sequences. Produce scores for the next word in the sequence.
    Scores should be raw logits not post-softmax activations.
    Load your model before generating predictions.
    :param inp: array of words (batch size, sequence length) [0-labels]
    :return: array of scores for the next word in each sequence (batch size, labels)
    """
    model_path = '../models/test.pt'
    chunk_path = model_path + '.npy.{}'  # format for each chunk
    data = chunks.read_chunks(chunk_path)
    # Load the data
    state_dict = chunks.load_from_numpy(data)
    model = models.LSTMModelV2(33278, 128, 256)
    # Load dictionary into your model
    model.load_state_dict(state_dict)
    model.eval()
    out = model(utils.to_variable(utils.to_tensor(inp))).data.numpy()[:,-1,:]
    return out
