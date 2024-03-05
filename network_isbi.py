
import torch
import os
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _pair
from transform import  PredictionTransform
# import resnext

def build_model(opt,in_frames, pred_dim, label_dim, image_points, tform_calib, tform_calib_R_T):
    """
    :param model_function: classification model
    """
    if opt.model_name == "efficientnet_b1":
        model = efficientnet_b1(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = pred_dim
        )
        # print(model.classifier[1].in_features)
        # print(model)



    elif opt.model_name == "efficientnet_b6":
        model = efficientnet_b6(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim
        )
    elif opt.model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels=in_frames,
            out_channels=model.features[0][0].out_channels,
            kernel_size=model.features[0][0].kernel_size,
            stride=model.features[0][0].stride,
            padding=model.features[0][0].padding,
            bias=model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=pred_dim
        )
    elif opt.model_name[:6] == "resnet":
        model = resnet101()
        model.conv1 = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.conv1.out_channels,
            kernel_size  = model.conv1.kernel_size,
            stride       = model.conv1.stride,
            padding      = model.conv1.padding,
            bias         = model.conv1.bias
        )
        model.fc = torch.nn.Linear(
            in_features   = model.fc.in_features,
            out_features  = pred_dim
        )


    elif opt.model_name == "LSTM_E":
        model = EncoderRNN_through_time(
            in_frames = in_frames,
            dim_feats = 1000,
            dim_hidden = 1024,
            n_layers = 1,
            bidirectional=False,
            input_dropout_p=0.2,
            rnn_cell='lstm',
            rnn_dropout_p=0.5,
            pred_dim = pred_dim)
    elif opt.model_name == "DCLNet50":
        model = resnext.resnet50(sample_size=2, sample_duration=16, cardinality=32,num_classes=pred_dim,input_ch=1)
        # model.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 7, 7),
        #                            stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, pred_dim)
    elif opt.model_name == "DCLNet101":
        model = resnext.resnet101(sample_size=2, sample_duration=16, cardinality=32,num_classes=pred_dim,input_ch=1)

    else:
        raise("Unknown model.")
    
    return model


# def Pretrained_model(in_frames):
#     pretrained_model = pretrainedmodels.resnet152(pretrained='imagenet')
#     pretrained_model.conv1 = torch.nn.Conv2d(
#         in_channels=in_frames,
#         out_channels=pretrained_model.conv1.out_channels,
#         kernel_size=pretrained_model.conv1.kernel_size,
#         stride=pretrained_model.conv1.stride,
#         padding=pretrained_model.conv1.padding,
#         bias=pretrained_model.conv1.bias
#     )
#     # pretrained_model.fc = torch.nn.Linear(
#     #     in_features=pretrained_model.fc.in_features,
#     #     out_features=pred_dim
#     # )
#
#     return pretrained_model

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='lstm'):
        """

        Args:
            dim_vid: dim of features of video frames
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden

class EncoderRNN_through_time(nn.Module):
    def __init__(self, in_frames,dim_feats, dim_hidden = 512, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='lstm', pred_dim = None):
        """
        input a feature of one frame at each time step, output all the transformations at the final time step
        Args:
            in_frames: input frames batchsize * number of images * h * w
            dim_feats: dim of features of frames
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN_through_time, self).__init__()
        self.in_frames = in_frames
        self.dim_feats = dim_feats
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell
        self.pred_dim = pred_dim
        # self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.out = nn.Linear(self.dim_hidden, self.pred_dim)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(self.dim_feats, self.dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        # use CNN to extract features for each image to reduce the dimention
        self.model_cnn = efficientnet_b1(weights=None)
        self.model_cnn.features[0][0] = torch.nn.Conv2d(
            in_channels=self.in_frames,
            out_channels=self.model_cnn.features[0][0].out_channels,
            kernel_size=self.model_cnn.features[0][0].kernel_size,
            stride=self.model_cnn.features[0][0].stride,
            padding=self.model_cnn.features[0][0].padding,
            bias=self.model_cnn.features[0][0].bias
        )
        self.model_cnn.classifier[1] = torch.nn.Linear(
            in_features=self.model_cnn.classifier[1].in_features,
            out_features=self.dim_feats*self.in_frames # the extracted feature map dimention is 1000; this parameter can try different experiment to setermine
        )

    #     self._init_hidden()
    #
    # def _init_hidden(self):
    #     nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, imgs):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        imgs_feats = self.model_cnn(imgs)
        imgs_feats = imgs_feats.reshape(imgs_feats.shape[0],imgs.shape[1],-1)
        # vid_feats = torch.squeeze(vid_feats,dim = 2) #.type(torch.FloatTensor)
        # batch_size, seq_len, dim_vid = (vid_feats).size()
        # hidden = self.vid2hid(vid_feats.view(-1, dim_vid))
        # hidden = self.input_dropout(hidden)
        # hidden = hidden.view(batch_size, seq_len, self.dim_hidden)
        hidden = None
        self.rnn.flatten_parameters()
        for i in range(imgs_feats.shape[1]):

            output, hidden = self.rnn(torch.unsqueeze(imgs_feats[:,i,:],dim=1),hidden)
            #  hidden - tuple; each tuple is 1*32*512
            #  output - 32*1*512
            #  hidden[0] == output
        # out = self.out(output.squeeze(1)), dim=1)
        out = torch.squeeze(self.out(output),dim=1)
        return out


class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)

    """

    def __init__(self,
                 pred_dim,
                 # max_len,
                 dim_hidden,
                 # dim_word,
                 n_layers=1,
                 rnn_cell='lstm',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output = pred_dim
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        # self.dim_word = dim_word
        # self.max_length = max_len
        # self.sos_id = 1
        # self.eos_id = 0
        self.input_dropout = nn.Dropout(input_dropout_p)
        # self.embedding = nn.Embedding(self.dim_output, dim_word)
        # self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                # opt={}
                ):
        """

        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences

        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        # sample_max = opt.get('sample_max', 1)
        # beam_size = opt.get('beam_size', 1)
        # temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        # seq_logprobs = []
        # seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs
            # targets_emb = self.embedding(targets)
            for i in range(self.dim_output):
                if i == 0:
                    current_transform = torch.LongTensor([self.sos_id] * batch_size).cuda()

                else:
                    current_transform = targets[:, i, :,:]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                # seq_logprobs.append(logprobs.unsqueeze(1))

            # seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            # if beam_size > 1:
            #     return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(
                    decoder_hidden.squeeze(0), encoder_outputs)

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                # elif sample_max:
                #     sampleLogprobs, it = torch.max(logprobs, 1)
                #     seq_logprobs.append(sampleLogprobs.view(-1, 1))
                #     it = it.view(-1).long()

                # else:
                    # sample according to distribuition
                    # if temperature == 1.0:
                    #     prob_prev = torch.exp(logprobs)
                    # else:
                    #     # scale logprobs by temperature
                    #     prob_prev = torch.exp(torch.div(logprobs, temperature))
                #     it = torch.multinomial(prob_prev, 1).cuda()
                #     sampleLogprobs = logprobs.gather(1, it)
                #     seq_logprobs.append(sampleLogprobs.view(-1, 1))
                #     it = it.view(-1).long()
                #
                # seq_preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)

            # seq_logprobs = torch.cat(seq_logprobs, 1)
            # seq_preds = torch.cat(seq_preds[1:], 1)

        return None

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h




def save_best_network(SAVE_PATH, model, epoch_label, gpu_ids):
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'saved_model', 'best_validation_model'))

    file_name = os.path.join(SAVE_PATH, 'opt.txt')
    with open(file_name, 'a') as opt_file:
        opt_file.write('------------ best validation result - epoch: -------------\n')
        opt_file.write(str(epoch_label))
        opt_file.write('\n')

# class EncoderCell(nn.Module):
#     def __init__(self,opt, pred_dim, label_dim, image_points, tform_calib, tform_calib_R_T):
#         super(EncoderCell, self).__init__()
#         self.opt = opt
#         # self.in_frames = in_frames
#         self.pred_dim = pred_dim
#         self.label_dim = label_dim
#         self.conv_0 = nn.Conv2d(
#             1, 64, kernel_size=3, stride=2, padding=1, bias=False) # self.in_frames
#
#         self.rnn1 = ConvLSTMCell(
#             64,
#             256,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             hidden_kernel_size=1,
#             bias=False)
#         self.rnn2 = ConvLSTMCell(
#             256,
#             512,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             hidden_kernel_size=1,
#             bias=False)
#         self.rnn3 = ConvLSTMCell(
#             512,
#             512,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#             hidden_kernel_size=1,
#             bias=False)
#
#         self.fc = torch.nn.Linear(
#             in_features=512*8*10,
#             out_features=self.pred_dim
#         )
#         self.fc_decoder = torch.nn.Linear(
#             in_features=512 * 8 * 10,
#             out_features=int(self.pred_dim/(self.in_frames-1))
#         )
#         self.transform_prediction = PredictionTransform(
#             opt.PRED_TYPE,
#             opt.LABEL_TYPE,
#             num_pairs=1,
#             image_points=image_points,
#             in_image_coords=True,
#             tform_image_to_tool=tform_calib,
#             tform_image_mm_to_tool=tform_calib_R_T
#         )
#
#     def forward(self, train_val, input, label, hidden1, hidden2, hidden3, device):
#         conv_in_decoder = nn.Conv2d(
#             int(self.label_dim / (input.shape[1] - 1)), 64, kernel_size=3, stride=2, padding=1, bias=False)
#         conv = nn.Conv2d(
#             input.shape[1], 64, kernel_size=3, stride=2, padding=1, bias=False)  # self.in_frames
#
#         if self.opt.model_name == 'LSTM_0':
#             x = conv(input)
#
#             hidden1 = self.rnn1(x, hidden1)
#             x = hidden1[0]
#
#             hidden2 = self.rnn2(x, hidden2)
#             x = hidden2[0]
#
#             hidden3 = self.rnn3(x, hidden3)
#             x = hidden3[0]
#
#             x = self.fc(x.view(x.size()[0], -1))
#         if self.opt.model_name == 'LSTM':
#             if self.opt.NUM_SAMPLES != -1:
#                 raise ("Inconsistent num_samples, should be -1 (sample all frames in a scan) for LSTM network.")
#             if self.opt.MINIBATCH_SIZE != -1:
#                 raise ("Inconsistent MINIBATCH_SIZE, should be 1 for LSTM network.")
#
#             for i in range(input.shape[1]):
#
#                 x = self.conv_0(input[:,i,:,:].unsqueeze(1))
#
#                 hidden1 = self.rnn1(x, hidden1)
#                 x = hidden1[0]
#
#                 hidden2 = self.rnn2(x, hidden2)
#                 x = hidden2[0]
#
#                 hidden3 = self.rnn3(x, hidden3)
#                 x = hidden3[0]
#
#             x = self.fc(x.view(x.size()[0], -1))
#
#         if self.opt.model_name == 'LSTM_GT':
#             # preds = torch.empty((input.shape[0],int(self.pred_dim/(self.in_frames-1))))
#             # encode part
#             for i in range(input.shape[1]):
#                 x = self.conv_0(input[:, i, :, :].unsqueeze(1))
#
#                 hidden1 = self.rnn1(x, hidden1)
#                 x = hidden1[0]
#
#                 hidden2 = self.rnn2(x, hidden2)
#                 x = hidden2[0]
#
#                 hidden3 = self.rnn3(x, hidden3)
#                 x = hidden3[0]
#
#             # decode part
#             gt_size = torch.empty(input.shape[0], label.shape[2]*label.shape[3], input.shape[2], input.shape[3])
#             for i in range(input.shape[1]-1):
#                 if i == 0:
#                     gt_f = torch.zeros_like(gt_size)
#                 else:
#                     if train_val == 'train':
#                         label_temp = (torch.reshape(label[:,i-1,:,:],(label.shape[0],-1)))
#                         gt_f = label_temp.unsqueeze(2).unsqueeze(3).repeat(1, 1, input.shape[2], input.shape[3])
#
#                     if train_val == 'val':
#                         pred_temp = torch.squeeze(self.transform_prediction(pred))
#                         pred_temp = (torch.reshape(pred_temp, (pred.shape[0], -1)))
#                         gt_f = pred_temp.unsqueeze(2).unsqueeze(3).repeat(1, 1, input.shape[2], input.shape[3])
#
#
#                 x = conv_in_decoder(gt_f.to(device))
#
#                 hidden1 = self.rnn1(x, hidden1)
#                 x = hidden1[0]
#
#                 hidden2 = self.rnn2(x, hidden2)
#                 x = hidden2[0]
#
#                 hidden3 = self.rnn3(x, hidden3)
#                 x = hidden3[0]
#
#                 pred = self.fc_decoder(x.view(x.size()[0], -1))
#
#                 if i == 0:
#                     preds = pred
#                 else:
#                     preds = torch.cat((preds, pred), dim=1)
#
#             x = preds
#
#
#
#
#
#         # x = self.conv(input)
#         #
#         # hidden1 = self.rnn1(x, hidden1)
#         # x = hidden1[0]
#         #
#         # hidden2 = self.rnn2(x, hidden2)
#         # x = hidden2[0]
#         #
#         # hidden3 = self.rnn3(x, hidden3)
#         # x = hidden3[0]
#         #
#         # x = self.fc(x.view(x.size()[0], -1))
#
#
#         return x, hidden1, hidden2, hidden3
#
# class ConvRNNCellBase(nn.Module):
#     def __repr__(self):
#         s = (
#             '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
#             ', stride={stride}')
#         if self.padding != (0, ) * len(self.padding):
#             s += ', padding={padding}'
#         if self.dilation != (1, ) * len(self.dilation):
#             s += ', dilation={dilation}'
#         s += ', hidden_kernel_size={hidden_kernel_size}'
#         s += ')'
#         return s.format(name=self.__class__.__name__, **self.__dict__)
#
#
# class ConvLSTMCell(ConvRNNCellBase):
#     def __init__(self,
#                  input_channels,
#                  hidden_channels,
#                  kernel_size=3,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  hidden_kernel_size=1,
#                  bias=True):
#         super(ConvLSTMCell, self).__init__()
#         self.input_channels = input_channels
#         self.hidden_channels = hidden_channels
#
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#
#         self.hidden_kernel_size = _pair(hidden_kernel_size)
#
#         hidden_padding = _pair(hidden_kernel_size // 2)
#
#         gate_channels = 4 * self.hidden_channels
#         self.conv_ih = nn.Conv2d(
#             in_channels=self.input_channels,
#             out_channels=gate_channels,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             dilation=self.dilation,
#             bias=bias)
#
#         self.conv_hh = nn.Conv2d(
#             in_channels=self.hidden_channels,
#             out_channels=gate_channels,
#             kernel_size=hidden_kernel_size,
#             stride=1,
#             padding=hidden_padding,
#             dilation=1,
#             bias=bias)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.conv_ih.reset_parameters()
#         self.conv_hh.reset_parameters()
#
#     def forward(self, input, hidden):
#         hx, cx = hidden
#         gates = self.conv_ih(input) + self.conv_hh(hx)
#
#         ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#
#         ingate = torch.sigmoid(ingate)
#         forgetgate = torch.sigmoid(forgetgate)
#         cellgate = torch.tanh(cellgate)
#         outgate = torch.sigmoid(outgate)
#
#         cy = (forgetgate * cx) + (ingate * cellgate)
#         hy = outgate * torch.tanh(cy)
#
#         return hy, cy
#
