import torch
import torch.nn as nn
from torch.nn import LSTM, LSTMCell
from torch.autograd import Variable

class EncoderLSTM(nn.Module):
    '''
        Based on encoder in Sequence2Sequence
        In the Encoder, we take in a sequence of inputs
        E.g. [Layer1, Layer2, Layer3...] or [Layer1.param1, Layer1.param2, Layer2.param1, ...]
        We return JUST the final hidden state after the LSTM has processed this sequence
        The idea is that the LSTM encodes the whole sequence in the hidden state
        We then pass the output of this to the Decoder to generate our sequence of actions
        
        NOTE: input to this is the reverse of sequence of inputs since its shown to perform better
    '''
    def __init__(self, input_size, hidden_size, seq_len, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        num_directions = 2 if bidirectional else 1
        
        self.lstm = LSTM(input_size, hidden_size, num_layers)
    
    def forward(self, input):
        # here input is some input sequence
        # define some initial hidden state
        hx = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)), Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        output, hx = self.lstm(input, hx)
        return hx 


class DecoderLSTM(nn.Module):
    '''
        Based on decoder in Sequence2Sequence
        In the Decoder, we take in a hidden state that is the output of the encoder
        Since this is autoregressive, we first generate an input to the LSTM which is the same size as the output
        This is done using the softmax layer
        Since the hidden states of the encoder and decoder are of the same dimension, this will work
        We then generate T outputs, where T is the length of our sequence (i.e. action)
        Since this is autoregressive, the input to each iteration is the softmax output of the previous one
    '''
    def __init__(self, output_size, hidden_size, seq_len, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super(DecoderLSTM, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.seq_len = seq_len
        num_directions = 2 if bidirectional else 1
        
        self.lstm = LSTMCell(output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
    
    def forward(self, hx):
        outputs = [] 
        # Convert hidden state of encoder into a input
        hx = (hx[0].squeeze(0), hx[1].squeeze(0))
        h = hx[0]
        input = self.softmax(self.linear(h))
        for i in range(self.seq_len):
            h, c = self.lstm(input, hx)
            hx = (h, c)
            # do softmax on output
            output = self.softmax(self.linear(h))
            outputs.append(output)
            input = output
        return torch.stack(outputs)


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, seq_len):
        super(EncoderDecoderLSTM, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, seq_len)
        self.decoder = DecoderLSTM(output_size, hidden_size, seq_len)
    
    def forward(self, input):
        h = self.encoder.forward(input)
        outputs = self.decoder.forward(h)
        return outputs

# from EncoderDecoder import * 
# from torch.autograd import Variable
# import torch
# ed = EncoderDecoderLSTM(5, 2, 100, 21)
# outputs = ed.forward(Variable(torch.rand(21, 1, 5)))
# actions = torch.stack([o.multinomial() for o in outputs])
