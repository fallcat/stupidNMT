'''
A module which implements the basic Transformer
'''
import uuid
import threading
import pdb
import torch
from torch import nn

from models.new_attention import NewAttention
from models.attention import MultiHeadedAttention
from models.embeddings import PositionEmbedding, TokenEmbedding
from models.utils import LabelSmoothingLoss, Translator
from utils import left_shift, right_shift, triu


class TransformerSublayer(nn.Module):
    '''
    Implements a sub layer of the transformer model, which consists of:
    1) A sub layer module
    2) Followed by dropout
    3) Plus a residual connection
    4) With layer normalization
    '''
    def __init__(self, sublayer, sublayer_shape, dropout_p=0.1):
        ''' Initialize the transformer sublayer '''
        super(TransformerSublayer, self).__init__()

        self.sublayer = sublayer
        self.norm = nn.LayerNorm(sublayer_shape)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        self.norm.reset_parameters()

    def forward(self, inputs, *sublayer_args, **sublayer_kwargs): # pylint:disable=arguments-differ
        ''' The forward pass of the sublayer '''
        return self.norm(inputs + self.dropout(self.sublayer(*sublayer_args, **sublayer_kwargs)))


class TransformerFFN(nn.Module):
    ''' Implements the Transformer feed-forward network '''
    def __init__(self, embedding_size, hidden_dim):
        super(TransformerFFN, self).__init__()

        self.relu = nn.ReLU()

        self.hidden = nn.Linear(embedding_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        ''' Reset parameters using xavier initialiation '''
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.hidden.weight, gain)
        nn.init.constant_(self.hidden.bias, 0.)

        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_uniform_(self.output.weight, gain)
        nn.init.constant_(self.output.bias, 0.)

    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' The forward pass of the feed-forward network '''
        return self.output(self.relu(self.hidden(inputs)))


class TransformerEncoderLayer(nn.Module):
    ''' Implements a single encoder layer in a transformer encoder stack '''
    def __init__(self, attn_config, num_heads, dim, hidden_dim, layer_i, dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerEncoderLayer, self).__init__()

        if attn_config['ffn_layer'][layer_i]:
            self.ffn = TransformerSublayer(
                TransformerFFN(dim, hidden_dim),
                dim, dropout_p
            )
            print('enc layer %i has ffn' % layer_i)

        self.self_attention = TransformerSublayer(
            NewAttention(attn_config, dim, num_heads),
            dim, dropout_p
        )

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()

    def forward(self, inputs, layer_i):  # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']

        # print("encoder self attention")

        state = self.self_attention(
            state,  # residual
            state, state, state, mask,  # passed to multiheaded attention
            layer_i=layer_i
        )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, # residual
                state # passed to feed-forward network
            )

        return {'state': state, 'mask': mask}


class TransformerDecoderLayer(nn.Module):
    ''' Implements a single decoder layer in a transformer decoder stack '''
    def __init__(self, dec_attn_config, enc_dec_attn_config, num_heads, dim, hidden_dim, layer_i, causal=True,
                 dropout_p=0.1):
        ''' Initialize the transformer layer '''
        super(TransformerDecoderLayer, self).__init__()

        self.causal = causal
        self.uuid = uuid.uuid4()

        self.enc_dec_attn_config = enc_dec_attn_config

        if dec_attn_config['ffn_layer'][layer_i]:
            self.ffn = TransformerSublayer(
                TransformerFFN(dim, hidden_dim),
                dim, dropout_p
            )
            print('dec layer %i has ffn' % layer_i)

        self.self_attention = TransformerSublayer(
            NewAttention(dec_attn_config, dim, num_heads),
            dim, dropout_p
        )

        if self.enc_dec_attn_config['enc_dec_attn_layer'] == 1 or \
                (type(self.enc_dec_attn_config['enc_dec_attn_layer'] is list) and
                 self.enc_dec_attn_config['enc_dec_attn_layer'][layer_i] == 1):
            if self.enc_dec_attn_config['enc_dec_attn_num_heads'] == -1:
                src_num_heads = num_heads
            elif type(self.enc_dec_attn_config['enc_dec_attn_num_heads']) is not list:
                src_num_heads = self.enc_dec_attn_config['enc_dec_attn_num_heads']
            else:
                src_num_heads = self.enc_dec_attn_config['enc_dec_attn_num_heads'][layer_i]
            assert src_num_heads != 0

            self.source_attention = TransformerSublayer(
                NewAttention(enc_dec_attn_config, dim, src_num_heads),
                dim, dropout_p
            )

            print('layer %i num of src heads %i' % (layer_i, src_num_heads))

    def reset_parameters(self):
        ''' Reset the parameters of the module '''
        self.ffn.reset_parameters()
        self.self_attention.reset_parameters()
        if hasattr(self, 'source_attention'):
            self.source_attention.reset_parameters()

    def forward(self, inputs, sources, layer_i): # pylint:disable=arguments-differ
        ''' The forward pass '''
        mask = inputs['mask']
        state = inputs['state']
        cache = inputs.get('cache')

        decoder_position = state.shape[1] - 1

        kwargs = {'layer_i': layer_i}
        if self.causal and cache is not None:
            # If caching, only want the last one sequence values. Requires no causal masking.
            residual = state[:, -1:]
            kwargs['decoder_position'] = decoder_position
        else:
            # If not caching, use the full sequence and ensure an appropriate causal mask
            residual = state
            kwargs['key_mask'] = mask
            kwargs['attention_mask'] = self.mask(state)

        # print("decoder self attention")
        state = self.self_attention(
            residual, # residual
            state, state, state, **kwargs # passed to multiheaded attention
        )

        source = sources['state']
        # print("source", source)
        kwargs = {'key_mask': sources['mask'], 'layer_i': layer_i}
        if self.causal and cache is not None:
            kwargs['decoder_position'] = decoder_position

        # print("decoder source attention")

        if hasattr(self, 'source_attention'):
            # print("in source, state", state.shape)
            state = self.source_attention(
                state, # residual
                source, source, state, **kwargs # passed to multiheaded attention
            )

        if hasattr(self, 'ffn'):
            state = self.ffn(
                state, # residual
                state # passed to feed-forward network
            )

        if self.causal and cache is not None:
            cached = cache.get(self.uuid)
            if cached is None:
                cache[self.uuid] = state
            else:
                # print("cached", cached.shape)
                # print("state", state.shape)
                try:
                    state = cache[self.uuid] = torch.cat((cached, state), 1)
                except:
                    pdb.set_trace()

        return {'state': state, 'mask': mask, 'cache': cache}

    _masks = threading.local()
    def mask(self, inputs):
        '''
        Get a self-attention mask
        The mask will be of shape [T x T] containing elements from the set {0, -inf}
        Input shape:  (B x T x E)
        Output shape: (T x T)
        '''
        if not self.causal:
            return None

        dim = inputs.shape[1]
        device = inputs.device
        mask_store = TransformerDecoderLayer._masks.__dict__
        if device not in mask_store:
            mask = inputs.new_full((dim, dim), float('-inf'))
            mask_store[device] = triu(mask, 1, 1, 1)

        mask = mask_store[device]
        if mask.shape[0] < dim:
            mask = mask.resize_(dim, dim).fill_(float('-inf'))
            mask_store[device] = triu(mask, 1, 1, 1)
            mask = mask_store[device]

        return mask[None, :dim, :dim]


class NewTransformer(nn.Module):
    ''' The New Transformer module '''
    def __init__(self, config, dataset):
        ''' Initialize the Transformer '''
        super(NewTransformer, self).__init__()

        self.dataset = dataset
        self.embedding = TokenEmbedding(
            dataset.vocab_size,
            config.embedding_size,
            padding_idx=self.padding_idx
        )
        self.position_embedding = PositionEmbedding(config.embedding_size)
        self.dropout = nn.Dropout(config.dropout_p, inplace=True)

        # Uniq attn attributes
        self.attn_ofs_uniq = list(set(
            config.enc_attn_offset + config.dec_attn_offset + config.enc_dec_attn_offset))
        self.attn_std_uniq = list(set(
            config.enc_attn_std + config.dec_attn_std + config.enc_dec_attn_std))

        # Allow for overriding the encoders and decoders in dervied classes
        self.encoders = self.create_encoders(config)
        self.decoders = self.create_decoders(config)

        self.label_smoothing = LabelSmoothingLoss(
            config.label_smoothing or 0,
            ignore_index=self.padding_idx,
            reduction='none'
        )
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction='none'
        )

    def create_encoders(self, config):
        ''' Create the transformer encoders '''
        kwargs = {'dropout_p': config.dropout_p}

        if config.ffn_layer == -1:
            config.ffn_layer = [1] * config.num_layers
        assert len(config.ffn_layer) == config.num_layers

        attn_config = {'attn_type': config.enc_attn_type,
                       'attn_std': config.enc_attn_std,
                       'attn_offset': config.enc_attn_offset,
                       'num_layers': config.num_layers,
                       'num_heads': config.num_heads,
                       'which_attn': 'encoder',
                       'attn_threshold': config.enc_attn_threshold,
                       'attn_window': config.enc_attn_window,
                       'attn_impl': config.enc_attn_impl,
                       'ffn_layer': config.ffn_layer,
                       'attn_ofs_uniq': self.attn_ofs_uniq,
                       'attn_std_uniq': self.attn_std_uniq}
        args = [attn_config, config.num_heads, config.embedding_size, config.hidden_dim]
        encoders = nn.ModuleList([
            TransformerEncoderLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        return encoders

    def create_decoders(self, config):
        ''' Create the transformer decoders '''
        kwargs = {'dropout_p': config.dropout_p}

        if config.ffn_layer == -1:
            config.ffn_layer = [1] * config.num_layers
        assert len(config.ffn_layer) == config.num_layers

        dec_attn_config = {'attn_type': config.dec_attn_type,
                           'attn_std': config.dec_attn_std,
                           'attn_offset': config.dec_attn_offset,
                           'num_layers': config.num_layers,
                           'num_heads': config.num_heads,
                           'which_attn': 'decoder',
                           'attn_threshold': config.dec_attn_threshold,
                           'attn_window': config.dec_attn_window,
                           'attn_impl': config.dec_attn_impl,
                           'ffn_layer': config.ffn_layer,
                           'attn_ofs_uniq': self.attn_ofs_uniq,
                           'attn_std_uniq': self.attn_std_uniq
                           }
        enc_dec_attn_config = {'attn_type': config.enc_dec_attn_type,
                               'attn_std': config.enc_dec_attn_std,
                               'attn_offset': config.enc_dec_attn_offset,
                               'num_layers': config.num_layers,
                               'num_heads': config.num_heads,
                               'word_count_ratio': self.dataset.word_count_ratio,
                               'which_attn': 'source',
                               'enc_dec_attn_layer': config.enc_dec_attn_layer,
                               'enc_dec_attn_num_heads': config.enc_dec_attn_num_heads,
                               'attn_threshold': config.enc_dec_attn_threshold,
                               'attn_window': config.enc_dec_attn_window,
                               'attn_impl': config.enc_dec_attn_impl,
                               'ffn_layer': config.ffn_layer,
                               'attn_ofs_uniq': self.attn_ofs_uniq,
                               'attn_std_uniq': self.attn_std_uniq
                               }
        args = [dec_attn_config, enc_dec_attn_config, config.num_heads, config.embedding_size, config.hidden_dim]
        decoders = nn.ModuleList([
            TransformerDecoderLayer(*args, layer_i, **kwargs)
            for layer_i in range(config.num_layers)
        ])

        return decoders


    @property
    def sos_idx(self):
        ''' Return the sos index '''
        return self.dataset.sos_idx

    @property
    def padding_idx(self):
        ''' Return the padding index '''
        return self.dataset.padding_idx

    def translator(self, config):
        ''' Get a translator for this model '''
        return Translator(config, self, self.dataset)

    def reset_named_parameters(self, modules):
        ''' Get a translator for this model '''
        if 'encoder' in modules:
            for encoder in self.encoders:
                encoder.reset_parameters()
        if 'decoder' in modules:
            for decoder in self.decoders:
                decoder.reset_parameters()
        if 'embeddings' in modules:
            self.embedding.reset_parameters()

    def forward(self, batch): # pylint:disable=arguments-differ
        ''' A batch of inputs and targets '''
        decoded = self.decode(
            self.encode(batch['inputs']),
            right_shift(batch['targets']),
            input_lens=batch['input_lens']
        )

        logits = decoded['logits']
        dims = list(range(1, logits.dim()))
        targets = left_shift(batch['targets'])
        nll = self.cross_entropy(logits, targets).sum(dims[:-1])
        smoothed_nll = self.label_smoothing(logits, targets).sum(dims)
        return smoothed_nll, nll

    def encode(self, inputs):
        ''' Encode the inputs '''
        word_embedding = self.embed(inputs, self.embedding)
        encoded = {
            'state': word_embedding,
            'mask': inputs.eq(self.padding_idx)
        }
        for i, encoder in enumerate(self.encoders):
            encoded = encoder(encoded, i)

        return encoded

    def decode(self, encoded, targets, decoders=None, embedding=None, cache=None, mask=None, input_lens=None):
        ''' Decode the encoded sequence to the targets '''
        if decoders is None:
            decoders = self.decoders

        if embedding is None:
            embedding = self.embedding

        word_embedding = self.embed(targets, embedding)

        decoded = {
            'cache': cache,
            'state': word_embedding,
            'mask': targets.eq(self.padding_idx) if mask is None else mask
        }
        for i, decoder in enumerate(decoders):
            # print("i", i)
            decoded = decoder(decoded, encoded, i)

        # compute projection to the vocabulary
        state = decoded['state']
        if cache is not None:
            state = state[:, -1:]

        return {
            'cache': decoded.get('cache'),
            'logits': embedding(state, transpose=True).transpose(2, 1),  # transpose to B x C x ...
        }

    def embed(self, inputs, token_embedding):
        ''' Embed the given inputs '''
        return self.dropout(token_embedding(inputs) + self.position_embedding(inputs))
