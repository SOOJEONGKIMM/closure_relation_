import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
import models.embedding as embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.utils import sort_batch_by_length, init_lstm, init_linear

class MLMAN(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100, drop=True):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.args = args
        self.conv = nn.Conv2d(1, self.hidden_size*2, kernel_size=(3, self.word_embedding_dim), padding=(1, 0))
        self.proj = nn.Linear(self.hidden_size*8, self.hidden_size)
        self.lstm_enhance = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.multilayer = nn.Sequential(nn.Linear(self.hidden_size*8, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))
        self.drop = drop
        self.dropout = nn.Dropout(0.2)
        self.cost = nn.CrossEntropyLoss()
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)
        # elif classname.find('Conv') != -1:
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, np.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).float())

    def context_encoder(self, input):
        input_mask = (input['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        embedding = self.embedding(input)
        embedding_ = embedding[:, :max_length].contiguous()

        if self.drop:
            embedding_ = self.dropout(embedding_)

        conv_out = self.conv(embedding_.unsqueeze(1)).squeeze(3)
        conv_out = conv_out * input_mask.unsqueeze(1)

        return conv_out.transpose(1, 2).contiguous(), input_mask, max_length

    def lstm_encoder(self, input, mask, lstm):
        if self.drop:
            input = self.dropout(input)
        mask = mask.squeeze(2)
        sequence_lengths = mask.long().sum(1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths.cpu(),
                                                     batch_first=True)
        lstmout, _ = lstm(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(lstmout, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor


    def CoAttention(self, support, query, support_mask, query_mask):
       # print("att: ", support.size(), query.size(), support_mask.size(), query_mask.size())

        att = support @ query.transpose(1, 2)
        att = att + support_mask * query_mask.transpose(1, 2) * 100
        support_ = F.softmax(att, 2) @ query * support_mask
        query_ = F.softmax(att.transpose(1,2), 2) @ support * query_mask
        return support_, query_

    def local_matching(self, support, query, support_mask, query_mask):

        support_, query_ = self.CoAttention(support, query, support_mask, query_mask)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)

        return enhance_support, enhance_query

    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query, support_mask, query_mask, K):

        max_enhance_query, _ = torch.max(enhance_query, 1)
        mean_enhance_query = torch.sum(enhance_query, 1) / torch.sum(query_mask, 1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)

        enhance_support = enhance_support.view(enhance_support.size(0) // K, K, -1, self.hidden_size * 2)
        support_mask = support_mask.view(enhance_support.size(0), K, -1, 1)

        max_enhance_support, _ = torch.max(enhance_support, 2)
        mean_enhance_support = torch.sum(enhance_support, 2) / torch.sum(support_mask, 2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)

        return enhance_support, enhance_query

    def forward(self, support, query, N, K, Q, eval=False):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''

        support, support_mask, support_len = self.context_encoder(support)
        query, query_mask, query_len = self.context_encoder(query)
        #print("encoded support shape:", support.size())
        #print("encoded query shape:", query.size())
        #print("support_mask shape:", support_mask.size())
        # print("encode support_s shape:", support_s.size())
        # print("support_mask_s shape:", support_mask_s.size())
        # print("query_mask shape:", query_mask.size())
        #print("support_len:", support_len)
        #print("query_len:", query_len)
        if not eval:
            if support.size(0) > N*K:
                support = support[-N*K:]
                support_mask = support_mask[-N*K:]
                #print("changed encoded support shape:", support.size())
                #print("changed support_mask shape:", support_mask.size())
        else:
            if support.size(0) > N*Q*K:
                support = support[-N*Q*K:]
                support_mask = support_mask[-N*Q*K:]
                #print("changed encoded support shape:", support.size())
                #print("changed support_mask shape:", support_mask.size())
        batch = support.size(0) // (N * K)
        #print("batch N Q K:", batch, N, Q, K)

        # concate S_k operation
        support = support.view(batch, 1, N, K, support_len, self.hidden_size * 2).expand(batch, N * Q, N, K,
                                                                                         support_len,
                                                                                         self.hidden_size * 2).contiguous().view(
            batch * N * Q * N, K * support_len, self.hidden_size * 2)
        support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N * Q, N, K,
                                                                             support_len).contiguous().view(-1,
                                                                                                            K * support_len,
                                                                                                            1)
        query = query.view(batch, N * Q, 1, query_len, self.hidden_size * 2).expand(batch, N * Q, N, query_len,
                                                                                    self.hidden_size * 2).contiguous().view(
            batch * N * Q * N, query_len, self.hidden_size * 2)
        query_mask = query_mask.view(batch, N * Q, 1, query_len).expand(batch, N * Q, N, query_len).contiguous().view(
            -1, query_len, 1)

        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        enhance_support = torch.relu(enhance_support)
        enhance_query = torch.relu(enhance_query)

        # split operation
        enhance_support = enhance_support.view(batch * N * Q * N * K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch * N * Q * N * K, support_len, 1)

        # LSTM
        enhance_support = self.lstm_encoder(enhance_support, support_mask, self.lstm_enhance)
        enhance_query = self.lstm_encoder(enhance_query, query_mask, self.lstm_enhance)

        # Local aggregation

        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask,
                                                                query_mask, K)

        tmp_query = enhance_query.unsqueeze(1).repeat(1, K, 1)
        cat_seq = torch.cat([tmp_query, enhance_support], 2)
        beta = self.multilayer(cat_seq)
        one_enhance_support = (enhance_support.transpose(1, 2) @ F.softmax(beta, 1)).squeeze(2)

        J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) ** 2, 2).mean()

        cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
        logits = self.multilayer(cat_seq)

        logits = logits.view(batch * N * Q, N)
        _, pred = torch.max(logits, 1)

        return logits, pred, J_incon
    def _forward(self,  support, query, N, K, Q, eval):#my support_snowball code
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        #print("support:",support)
        #print("query:",query.keys())

        support, support_mask, support_len = self.context_encoder(support)
        #support_s, support_mask_s, support_len_s = self.context_encoder(support_snowball)
        query, query_mask, query_len = self.context_encoder(query)
        print("support:", support.size(), "supportmask",support_mask.size())

        print("encoded support shape:", support.size())
        print("encoded query shape:", query.size())
        print("support_mask shape:", support_mask.size())
       #print("encode support_s shape:", support_s.size())
        #print("support_mask_s shape:", support_mask_s.size())
        #print("query_mask shape:", query_mask.size())
        print("support_len:", support_len)
        #print("query_len:", query_len)

        batch = support.size(0)//(N*K)
#        print("B N K Q:", batch, N, K, Q)

        q_batch = batch #100 40 20, 100 40 맞추려면 배치사이즈 1
        if support.size(1)<support_len:
            leftover = support_len - support.size(1)
            support = torch.cat((support, support[:,:leftover, :]), 1)
            support_mask = torch.cat((support_mask, support_mask[:,:leftover]), 1)
            support_len = support.size(1)

        if support.size(0)<(N*K):
            x = (N*K) // support.size(0)
            support = support.repeat(x,1,1)
          #  print("batch, support.size(0)%N*K:", batch, support.size(0)%(N*K))
            leftover = (N*K) - support.size(0)
         #   print("leftover:", leftover)
            if leftover>0:
                support = torch.cat((support, support[:leftover,:,:]), 0)
             #   print("new support size:", support.size())
            x = (N*K) // support_mask.size(0)
            mask_leftover = (N*K) - support_mask.size(0)
            support_mask = support_mask.repeat(x, 1)
         #   print("mask_leftover:", mask_leftover)
            if mask_leftover>0:
               # fill_tensor = torch.zeros((mask_leftover, support_mask.size(1))).cuda()
                support_mask = torch.cat((support_mask, support_mask[:leftover,:]), 0)
          #      print("new support_mask size:", support_mask.size())
        else:
            q_batch = query.size(0) // (N * K)

            #print("B N K Q:", batch, N, K, Q)
            if not eval:
                support = support[:N*K,:,:]
                support_mask = support_mask[:N*K,:]
                batch = support.size(0) // (N * K)
                if q_batch*(N*Q)>query.size(0):
                    q_batch = query.size(0) // (N * Q)
            else:
                support = support[:N * Q, :, :]
                support_mask = support_mask[:N * Q, :]
                batch = support.size(0) // (N * K)
                q_batch = query.size(0) // (N * Q)


        if not eval:
            if query.size(0)<(N*Q):
                x = (N*Q) // query.size(0)
                query = query.repeat(x,1,1)
             #   print("batch, support.size(0)%N*K:", batch, query.size(0)%(N*K))
                leftover = (N*Q) - query.size(0)
            #    print("leftover:", leftover)
                if leftover>0:
                    query = torch.cat((query, query[:leftover,:,:]), 0)
                  #  print("new query size:", query.size())
                x = (N*Q) // query_mask.size(0)
                mask_leftover = (N*Q) - query_mask.size(0)
                query_mask = query_mask.repeat(x, 1)
            #    print("mask_leftover:", mask_leftover)
                if mask_leftover>0:
                   # fill_tensor = torch.zeros((mask_leftover, support_mask.size(1))).cuda()
                    query_mask = torch.cat((query_mask, query_mask[:leftover,:]), 0)
               #     print("new query_mask size:", query_mask.size())
            else:
                query = query[:N*Q,:,:]
                query_mask = query_mask[:N*Q,:]

            if query.size(1) < query_len:
                leftover = query_len - query.size(1)
                query = torch.cat((query, query[:, :leftover, :]), 1)
                query_mask = torch.cat((query_mask, query_mask[:, :leftover]), 1)
                query_len = query.size(1)



       # print("reshaped support shape:", support.size())
       # print("reshaped support mask shape:", support_mask.size())
        #print("reshaped query shape:", query.size())
        #print("reshaped query mask shape:", query_mask.size())

        # concate S_k operation


        #support = support.view(batch, 1, N, K, support_len, self.hidden_size*2).expand(batch, N*Q, N, K, support_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, K*support_len, self.hidden_size*2)
        #support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N*Q, N, K, support_len).contiguous().view(-1, K*support_len, 1)
        if not eval:
            '''
            support = support.view(q_batch, N * Q, 1, support_len, self.hidden_size * 2).expand(q_batch, N * Q, N, support_len,
                                                                                          self.hidden_size * 2).contiguous().view(
                q_batch * N * Q * N, support_len, self.hidden_size * 2)  # N빼고 Q넣음
            support_mask = support_mask.view(q_batch, N * Q, 1, support_len).expand(q_batch, N * Q, N,
                                                                              support_len).contiguous().view(
                -1, support_len, 1)  # N빼고 Q넣음
                '''
            support = support.view(batch, 1, N, K, support_len, self.hidden_size * 2).expand(batch, N * Q, N, K,
                                                                                             support_len,
                                                                                             self.hidden_size * 2).contiguous().view(
                batch * N * Q * N, K * support_len, self.hidden_size * 2)
            support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N * Q, N, K,
                                                                                 support_len).contiguous().view(-1,
                                                                                                                K * support_len,
                                                                                                                1)
            query = query.view(batch, N * Q, 1, query_len, self.hidden_size * 2).expand(batch, N * Q, N, query_len,
                                                                                        self.hidden_size * 2).contiguous().view(
                batch * N * Q * N, query_len, self.hidden_size * 2) #N빼고 Q넣음

            query_mask = query_mask.view(batch, N * Q, 1, query_len).expand(batch, N * Q, N, query_len).contiguous().view(
                -1, query_len, 1) #N빼고 Q넣음
        elif eval:
            #batch = support.size(0) // (N * K)
            #q_batch = batch
            support = support.view(batch, 1, N, K, support_len, self.hidden_size * 2).expand(batch, N * Q, N, K,
                                                                                             support_len,
                                                                                             self.hidden_size * 2).contiguous().view(
                batch * N * Q * N, K * support_len, self.hidden_size * 2)
            support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N * Q, N, K,
                                                                                 support_len).contiguous().view(-1,
                                                                                                                K * support_len,
                                                                                                                1)
            query = query.view(q_batch, N * Q, 1, query_len, self.hidden_size * 2).expand(q_batch, N * Q, N, query_len,
                                                                                        self.hidden_size * 2).contiguous().view(
                q_batch * N * Q * N, query_len, self.hidden_size * 2)
            query_mask = query_mask.view(q_batch, N * Q, 1, query_len).expand(q_batch, N * Q, N,
                                                                            query_len).contiguous().view(
                -1, query_len, 1)
        #print("-----------", support.size(), support_mask.size(), query.size(), query_mask.size()) #2000 40 200, 2000 40 1
        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)
        #print("enhance_support:", enhance_support.size(), "enhance_query:", enhance_query.size())

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        enhance_support = torch.relu(enhance_support)
        enhance_query = torch.relu(enhance_query)

        # split operation
        enhance_support = enhance_support.view(batch * N * Q * N * K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch * N * Q * N * K, support_len, 1)

        # LSTM
        enhance_support = self.lstm_encoder(enhance_support, support_mask, self.lstm_enhance)
        enhance_query = self.lstm_encoder(enhance_query, query_mask, self.lstm_enhance)

        # Local aggregation

        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask,
                                                                query_mask, K)

        tmp_query = enhance_query.unsqueeze(1).repeat(1, K, 1)
        cat_seq = torch.cat([tmp_query, enhance_support], 2)
        beta = self.multilayer(cat_seq)
        one_enhance_support = (enhance_support.transpose(1, 2) @ F.softmax(beta, 1)).squeeze(2)
   #     print("one_enhance_support:", one_enhance_support.size())
    #    print("enhance_support:", enhance_support.size())
  #      print("enhance_query:", enhance_query.size())
  #      print("beta:", beta.size())

        J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) ** 2, 2).mean()
   #     print("J_incon:", J_incon)


        cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
        logits = self.multilayer(cat_seq)
   #     print("Logits:", logits.size()) #100 20

        if not eval:
            logits = logits.view(5 * N, N) #batch를 5로 바꿈
        elif eval:
            logits = logits.view(batch * N * Q, N)
      #  print("2Logits:", logits.size())
        _, pred = torch.max(logits, 1)
    #    print("pred:", pred.size())

        return logits, pred, J_incon
