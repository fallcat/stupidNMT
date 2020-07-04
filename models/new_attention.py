'''
A module which implements various attention mechanisms
'''
import pdb
import math
import torch
import threading
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

from utils import same_tensor

class NewAttention(nn.Module):
    ''' Implement a hard-coded attention module '''

    def __init__(self, attn_config, embed_dim, num_heads=1):
        ''' Initialize the attention module '''
        super(NewAttention, self).__init__()

        # ensure valid inputs
        assert embed_dim % num_heads == 0, \
            f'num_heads={num_heads} should evenly divide embed_dim={embed_dim}'

        # store off the scale and input params
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = attn_config['num_layers']
        self.projection_dim = embed_dim // num_heads
        self.scale = self.projection_dim ** -0.5
        self.output_projection = nn.Linear(embed_dim, embed_dim, bias=False)

        # hard-coded attn param
        self.attn_type = attn_config['attn_type']               # learned / normal
        self.attn_offset = attn_config['attn_offset']
        self.attn_std = attn_config['attn_std']
        self.attn_threshold = attn_config['attn_threshold']

        # conv param
        self.attn_window = attn_config['attn_window']
        self.half_window = int((self.attn_window - 1) / 2)

        # average word count ratio between two languages
        self.word_count_ratio = attn_config['word_count_ratio'] if 'word_count_ratio' in attn_config else 1

        # attn implementation param
        self.attn_impl = attn_config['attn_impl']               # full, conv, index

        # Combine projections for multiple heads into a single linear layer for efficiency
        self.input_weights = None
        if 'learned' in self.attn_type or 'learned' == self.attn_type:
            self.input_weights = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        else:
            self.input_weights = nn.Parameter(torch.Tensor(embed_dim, embed_dim))

        self.reset_parameters()
        self.attn_configs = list(self.load_attn_configs())

        self.max_absolute_offset = max([abs(a) for a in attn_config['attn_offset']])
        self.attn_ofs_uniq = attn_config['attn_ofs_uniq']
        self.attn_std_uniq = attn_config['attn_std_uniq']

        self.impl = attn_config['attn_impl']

    _attn_indices = threading.local()

    def get_attn_indices(self, qlen, attn_offset, device):

        attn_idx_store = NewAttention._attn_indices.__dict__

        if device not in attn_idx_store:
            indices_q = torch.arange(self.max_absolute_offset, self.max_qlen + self.max_absolute_offset).view(1, -1)
            attn_ofs_uniq = torch.tensor(self.attn_ofs_uniq).view(-1, 1)
            attn_idx_store[device] = (indices_q + attn_ofs_uniq).to(device)
        offset_idx = [self.attn_ofs_uniq.index(i) for i in attn_offset]
        return attn_idx_store[device][offset_idx, :qlen][None, :, :, None]  # 1, nh, qlen, 1

    _attn_cache = threading.local()

    def get_attn_cache(self, attn_std, attn_offset, qlen, vlen, device, decoder_position=-1):

        attn_cache_store = NewAttention._attn_cache.__dict__

        real_qlen = max(qlen, decoder_position + 1)

        max_offset, min_offset = max(self.attn_ofs_uniq), min(self.attn_ofs_uniq)

        if device not in attn_cache_store or attn_cache_store[device].shape[1] < real_qlen or attn_cache_store[device].shape[2] < vlen:
            max_qlen = max(attn_cache_store[device].shape[1], real_qlen) if device in attn_cache_store else real_qlen
            max_vlen = max(attn_cache_store[device].shape[2], vlen) if device in attn_cache_store else vlen

            attn_std_uniq = torch.tensor(self.attn_std_uniq).view(-1, 1, 1)
            indices_q = torch.arange(max_qlen).float().view(1, -1, 1) * self.word_count_ratio
            indices_v = torch.arange(-max_offset, max_vlen - min_offset).float().view(1, 1, -1)  # -max_offset: focus on right most position, self.max_qlen - min_offset: leftmost

            distance_diff = indices_v - indices_q
            logits = (1 / (attn_std_uniq * math.sqrt(2 * math.pi)) * torch.exp(
                - 1 / 2 * (distance_diff / attn_std_uniq) ** 2))
            if self.attn_threshold > 0:
                logits[logits < self.attn_threshold] = 0

            attn_cache_store[device] = logits.to(device)

        std_idx = [self.attn_std_uniq.index(i) for i in attn_std]
        attn_ofs_l = np.array([max_offset - a for a in attn_offset])
        attn_ofs_r = np.array([max_offset - a + vlen for a in attn_offset])

        retrieved = attn_cache_store[device]  # nh x qlen x vlen
        retrieved = retrieved[
            [[[a]] for a in std_idx], [[[b] for b in list(range(real_qlen))]], [[list(range(l, r))] for l, r in
                                                                           zip(attn_ofs_l, attn_ofs_r)]]
        if decoder_position == -1:
            return retrieved[:, :qlen, :vlen]

        else:
            return retrieved[:, decoder_position:decoder_position+1, :vlen]

    def reset_parameters(self):
        ''' Reset parameters using xavier initialization '''
        # Initialize using Xavier
        gain = nn.init.calculate_gain('linear')
        if self.input_weights is not None:
            nn.init.xavier_uniform_(self.input_weights, gain)
        nn.init.xavier_uniform_(self.output_projection.weight, gain)

    def project(self, inputs, index=0, chunks=1, project=True):
        ''' Produce a linear projection using the weights '''
        batch_size = inputs.shape[0]
        start = index * self.embed_dim
        end = start + chunks * self.embed_dim

        if project:
            projections = F.linear(inputs, self.input_weights[start:end]).chunk(chunks, dim=-1)
        else:
            projections = [inputs]

        output_projections = []
        for projection in projections:
            # transform projection to (BH x T x E)
            output_projections.append(
                projection.view(
                    batch_size,
                    -1,
                    self.num_heads,
                    self.projection_dim
                ).transpose(2, 1).contiguous().view(
                    batch_size * self.num_heads,
                    -1,
                    self.projection_dim
                )
            )

        return output_projections

    def project_learned(self, inputs, learned_idx):
        batch_size = int(inputs.shape[0] / self.num_heads)
        return inputs.view(batch_size,
                           self.num_heads,
                           -1,
                           self.projection_dim)[:, learned_idx].contiguous()\
            .view(batch_size * len(learned_idx),
                  -1,
                  self.projection_dim)

    def load_attn_configs(self):

        """
            expand attn_configs for each layer

            c: input attn_configs; h: head; l: layer
            len(c) == 1 :         all attn heads are the same
            len(c) == #h :        each layer use the same head combinations
            len(c) == #l:         each layer uses same config for each head, different layers use different attn_configs
            len(c) == #h x #l :   all attn_configs specified, layer_0_head_0, layer_0_head_1, ..., layer_(l-1)_head_(h-1)
            #heads % len(c) == 0: repeat head attn_configs to #heads for each layer

        """

        for li in range(self.num_layers):

            attn_configs = {}

            for attr in self.__dict__:
                if not attr.startswith('attn_'):
                    continue

                c = getattr(self, attr)
                if type(c) is not list:
                    c = [c] * self.num_heads
                else:

                    if len(c) == self.num_layers * self.num_heads:
                        c = c[li * self.num_heads: (li + 1) * self.num_heads]

                    elif len(c) == self.num_heads:
                        c = c

                    elif len(c) == self.num_layers:
                        c = [c[li]] * self.num_heads

                    elif self.num_heads % len(c) == 0:
                        c *= self.num_heads // len(c)

                    else:
                        raise ValueError('wrong head attn_configs')

                attn_configs[attr] = c

            if self.attn_window == -1:
                yield attn_configs, None
                continue

            with torch.no_grad():

                distance_diff = torch.arange(-self.half_window, self.half_window + 1, dtype=torch.float32,
                                             device=torch.device("cuda"))
                # conv_filter: (self.window_size,)

                conv_filters = {}  # conv_filters[std] stores conv filters
                masked_conv_filters = {} # conv_filters[std][offset] stores masked conv filters

                attn_std, attn_offset = attn_configs['attn_std'], attn_configs['attn_offset']
                head_configs = defaultdict(list)
                for i, c in enumerate(zip(attn_std, attn_offset)):
                    head_configs[c].append(i)

                for hc, idx in head_configs.items():
                    attn_std, attn_offset = hc[0], hc[1]
                    conv_filter = (1 / (attn_std * math.sqrt(2 * math.pi)) * torch.exp(
                        - 1 / 2 * (distance_diff / attn_std) ** 2)).view(1, 1, -1)
                    conv_filters[attn_std] = conv_filter
                    conv_filter[self.attn_window - (self.half_window + attn_offset):] = 0
                    masked_conv_filters[attn_std][attn_offset] = conv_filter

            yield attn_configs, conv_filters, masked_conv_filters

    def mha_reshape(self, tensor, batch_size):
        '''
            multi-headed attention reshape
            tensor.shape = B*H x L x proj_dim
            output tensor.shape = B x L x E
        '''

        return tensor.view(batch_size, self.num_heads, -1,
                           self.projection_dim).transpose(2, 1).contiguous().view(
                            batch_size, -1, self.num_heads * self.projection_dim
                           )

    def gather_reshape(self, tensor, attn_indices, bsz, qlen, dp):
        '''
            used in `conv` and `indexing` implementation
            dp: decoder position
        '''
        if dp == -1:
            return torch.gather(tensor, 2, attn_indices.expand(
                    bsz, self.num_heads, qlen, self.projection_dim)
                    ).transpose(2,1).contiguous().view(bsz, -1, self.num_heads * self.projection_dim)
        else:
            return torch.gather(tensor, 2, attn_indices[:, :, dp:dp+1].expand(
                    bsz, self.num_heads, qlen, self.projection_dim)
                    ).transpose(2,1).contiguous().view(bsz, -1, self.num_heads * self.projection_dim)

    def attention_index(self, values, keys, queries, key_mask=None, mask=None, layer_i=0, decoder_position=-1):
        queries_shape = queries.shape  # B*H x L x E
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, _ = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs[
            'attn_offset']

        # bs x num_heads x vlen x proj_dim
        values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])

        if key_mask is not None:
            values.masked_fill_(key_mask[:, None, :, None], float(0))

        values = F.pad(values, (0, 0, self.max_absolute_offset, self.max_absolute_offset), "constant", 0)
        # recompute attended indices
        attn_indices = self.get_attn_indices(max(queries_shape[1], decoder_position + 1), attn_offset, values.device)

        return self.gather_reshape(values, attn_indices, batch_size, queries_shape[1], decoder_position)

    def attention_conv(self, values, keys, queries, key_mask=None, mask=None, layer_i=0, decoder_position=-1):

        queries_shape = queries.shape  # B*H x L x proj_dim
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, conv_filters = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs[
            'attn_offset']

        curr_conv_filter = []
        for i in range(self.num_heads):
            curr_conv_filter.append(conv_filters[attn_std[i]][attn_offset[i]])
        curr_conv_filter = torch.cat(curr_conv_filter, dim=0)

        values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])

        if key_mask is not None:
            values.masked_fill_(key_mask[:, None, :, None], float(0))

        values = values.transpose(3, 1).transpose(3, 2).contiguous().view(batch_size * self.projection_dim,
                                                                          self.num_heads, -1)
        attended = F.conv1d(values, curr_conv_filter, padding=self.half_window + self.max_absolute_offset,
                            groups=self.num_heads)
        attended = attended.view(batch_size, self.projection_dim, self.num_heads, -1).transpose(1, 2).transpose(2,
                                                                                                                3).contiguous()

        # recompute attended indices
        attn_indices = self.get_attn_indices(max(queries_shape[1], decoder_position + 1), attn_offset, values.device)

        return self.gather_reshape(attended, attn_indices, batch_size, queries_shape[1], decoder_position)

    def compute_together(self, attn_type, attn_std, attn_offset):

        return len(set(attn_type)) == 1 and \
               len(set(attn_std)) == 1 and \
               len(set(attn_offset)) == 1

    def attention(self, values, keys, queries, key_mask=None, mask=None, layer_i=0, decoder_position=-1):

        queries_shape = queries.shape  # B*H x L x proj_dim
        values_shape = values.shape
        batch_size = queries_shape[0] // self.num_heads

        attn_configs, _ = self.attn_configs[layer_i]
        attn_type, attn_std, attn_offset = attn_configs['attn_type'], attn_configs['attn_std'], attn_configs[
            'attn_offset']

        if all(a == 'learned' for a in attn_type):  # all heads are learned

            logits = self.scale * torch.bmm(queries, keys.transpose(2, 1))

            if mask is not None:
                logits += mask

            if key_mask is not None:
                logits_shape = logits.shape
                batch_size = logits_shape[0] // self.num_heads
                logits = logits.view(batch_size, self.num_heads, logits_shape[1], logits_shape[2])
                logits.masked_fill_(key_mask[:, None, None], float('-inf'))
                logits = logits.view(logits_shape)

            attn_weights = F.softmax(logits, dim=-1)

            attended = torch.bmm(attn_weights, values)

            return self.mha_reshape(attended, batch_size)

        elif 'learned' in attn_type:

            learned_indices = [i for i, x in enumerate(attn_type) if x == 'learned']
            queries_ = self.project_learned(queries, learned_indices)
            keys_ = self.project_learned(keys, learned_indices)

            logits_ = self.scale * torch.bmm(queries_, keys_.transpose(2, 1))
            logits_shape_ = logits_.shape

            if mask is not None:
                logits_ += mask

            if key_mask is not None:
                batch_size = logits_shape_[0] // len(learned_indices)
                logits_ = logits_.view(batch_size, len(learned_indices), logits_shape_[1], logits_shape_[2])
                logits_.masked_fill_(key_mask[:, None, None], float('-inf'))
                logits_ = logits_.view(logits_shape_)

            logits_ = F.softmax(logits_, dim=-1).view(batch_size,
                                                      len(learned_indices),
                                                      logits_shape_[-2],
                                                      logits_shape_[-1])

        with torch.no_grad():

            # if config for all heads in the same layer is the same, compute them together
            if self.compute_together(attn_type, attn_std, attn_offset):

                attn_type = attn_type[0]
                attn_std = [attn_std[0]]
                attn_offset = [attn_offset[0]]

                logits = self.get_attn_cache(attn_std, attn_offset, queries_shape[1], values_shape[1], values.device,
                                             decoder_position=decoder_position)

                # Copy the weights to each head
                attn_weights = logits.expand(batch_size, self.num_heads, queries_shape[1], values_shape[1]) \
                    .contiguous().view(-1, queries_shape[1], values_shape[1])

            # if not all heads have the same config
            else:

                logits_list = []

                hc_indices = [i for i, t in enumerate(attn_type) if t != 'learned']
                attn_std = [x for i, x in enumerate(attn_std) if i in hc_indices]
                attn_offset = [x for i, x in enumerate(attn_offset) if i in hc_indices]
                logits = self.get_attn_cache(attn_std, attn_offset, queries_shape[1], values_shape[1], values.device,
                                             decoder_position=decoder_position)
                attn_weights = values.new_zeros(batch_size, self.num_heads, queries_shape[1], values_shape[1])
                try:
                    attn_weights[:, hc_indices] = logits.expand(batch_size, len(hc_indices), queries_shape[1],
                                                            values_shape[1])
                except:
                    pdb.set_trace()
                if 'learned' in attn_type:
                    attn_weights[:, learned_indices] = logits_  # bs x learned_indices x L x L

                # print("attn_weights", attn_weights)

                attn_weights = attn_weights.contiguous().view(-1, queries_shape[1], values_shape[1])

        if mask is not None:
            attn_weights = attn_weights * (mask == 0).to(dtype=torch.float32)

        if key_mask is not None:
            # bs x num_heads x vlen x proj_dim
            values = values.view(batch_size, self.num_heads, values_shape[1], values_shape[2])
            values.masked_fill_(key_mask[:, None, :, None], float(0))
            values = values.view(values_shape)

        attended = torch.bmm(attn_weights, values)

        return self.mha_reshape(attended, batch_size)

    def forward(self, values, keys, queries, # pylint:disable=arguments-differ
                key_mask=None, attention_mask=None, layer_i=0, decoder_position=-1):
        ''' Forward pass of the attention '''
        batch_size = values.shape[0]
        # print("key_mask", key_mask)

        if 'learned' in self.attn_type or 'learned' == self.attn_type:
            if same_tensor(values, keys, queries):
                values, keys, queries = self.project(values, chunks=3)
            elif same_tensor(values, keys):
                values, keys = self.project(values, chunks=2)
                queries, = self.project(queries, 2)
            else:
                values, = self.project(values, 0)
                keys, = self.project(keys, 1)
                queries, = self.project(queries, 2)

        else:
            values, = self.project(values, 0, project=True)
            keys, = self.project(keys, 1, project=False)
            queries, = self.project(queries, 2, project=False)
        # pylint:enable=unbalanced-tuple-unpacking

        if decoder_position != -1:
            queries = queries[:, -1:]

        if 'full' in self.impl:
            attended = self.attention(values, keys, queries,
                                  key_mask, attention_mask, layer_i, decoder_position)

        elif 'conv' in self.impl:
            attended = self.attention_conv(values, keys, queries,
                                  key_mask, attention_mask, layer_i, decoder_position)

        elif 'index' in self.impl:
            attended = self.attention_index(values, keys, queries,
                                  key_mask, attention_mask, layer_i, decoder_position)

        return self.output_projection(attended)
