from modules.tts_modules import TransformerEncoderLayer, TransformerDecoderLayer, \
    DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from modules.operations import *


class TransformerEncoder(nn.Module):
    def __init__(self, arch, embed_tokens, last_ln=True):
        super().__init__()
        self.arch = arch
        self.num_layers = hparams['enc_layers']
        self.hidden_size = hparams['hidden_size']
        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        embed_dim = embed_tokens.embedding_dim
        self.dropout = hparams['dropout']
        self.embed_scale = math.sqrt(embed_dim)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        positions = self.embed_positions(src_tokens)
        # x = self.prenet(x)
        x = embed + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens):
        """

        :param src_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
            'encoder_padding_mask': [B x T]
            'encoder_embedding': [B x T x C]
            'attn_w': []
        }
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx).data

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'attn_w': []
        }


class TransformerDecoder(nn.Module):
    def __init__(self, arch, padding_idx=0, num_layers=None, causal=True, dropout=None, out_dim=None):
        super().__init__()
        self.arch = arch
        self.num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        self.hidden_size = hparams['hidden_size']
        self.prenet_hidden_size = hparams['prenet_hidden_size']
        self.padding_idx = padding_idx
        self.causal = causal
        self.dropout = hparams['dropout'] if dropout is None else dropout
        self.in_dim = hparams['audio_num_mel_bins']
        self.out_dim = hparams['audio_num_mel_bins'] + 1 if out_dim is None else out_dim
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_target_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(self.arch[i], self.hidden_size, self.dropout)
            for i in range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)
        self.project_out_dim = Linear(self.hidden_size, self.out_dim, bias=False)
        self.prenet_fc1 = Linear(self.in_dim, self.prenet_hidden_size)
        self.prenet_fc2 = Linear(self.prenet_hidden_size, self.prenet_hidden_size)
        self.prenet_fc3 = Linear(self.prenet_hidden_size, self.hidden_size, bias=False)

    def forward_prenet(self, x):
        mask = x.abs().sum(-1, keepdim=True).ne(0).float()

        prenet_dropout = 0.5
        # prenet_dropout = random.uniform(0, 0.5) if self.training else 0
        x = self.prenet_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, prenet_dropout, training=True)
        x = self.prenet_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, prenet_dropout, training=True)
        x = self.prenet_fc3(x)
        x = F.relu(x)
        x = x * mask
        return x

    def forward(
            self,
            prev_output_mels,  # B x T x 80
            encoder_out=None,  # T x B x C
            encoder_padding_mask=None,  # B x T x C
            target_mels=None,
            incremental_state=None,
    ):
        # embed positions
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_output_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            prev_output_mels = prev_output_mels[:, -1:, :]
            positions = positions[:, -1:, :]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                target_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            self_attn_padding_mask = target_mels.abs().sum(-1).eq(0).data

        # convert mels through prenet
        x = self.forward_prenet(prev_output_mels)
        # embed positions
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        all_attn_logits = []

        # decoder layers
        for layer in self.layers:
            if incremental_state is None and self.causal:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask
            )
            all_attn_logits.append(attn_logits)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x C -> B x T x 81
        x = self.project_out_dim(x)

        # attn_logits = torch.stack(all_attn_logits, dim=1) # B x n_layers x head x target_len x src_len
        return x, all_attn_logits

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(tts_utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerTTS(nn.Module):
    def __init__(self, arch, dictionary, num_spk=0, causal_decoder=True):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        self.enc_layers = hparams['enc_layers']
        self.dec_layers = hparams['dec_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers + self.dec_layers]
        self.hidden_size = hparams['hidden_size']
        self.mel = hparams['audio_num_mel_bins']
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.causal_decoder = causal_decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        if num_spk > 0:
            self.spk_embed = nn.Embedding(num_spk, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def build_encoder(self):
        return TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)

    def build_decoder(self):
        return TransformerDecoder(self.dec_arch, padding_idx=self.padding_idx,
                                  causal=self.causal_decoder)

    def forward_encoder(self, src_tokens, spk_ids=None, *args, **kwargs):
        return self.encoder(src_tokens)

    def forward_decoder(self, prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=None):
        decoder_output, attn_logits = self.decoder(
            prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=incremental_state)
        return decoder_output, attn_logits

    def forward(self, src_tokens, prev_output_mels, target_mels, spk_ids=None, *args, **kwargs):
        encoder_outputs = self.forward_encoder(src_tokens, spk_ids)
        encoder_out = encoder_outputs['encoder_out']
        if spk_ids is not None:
            encoder_out += self.spk_embed(spk_ids)[None, :, :]
        encoder_padding_mask = encoder_outputs['encoder_padding_mask'].data
        decoder_output, attn_logits = self.decoder(prev_output_mels, encoder_out, encoder_padding_mask,
                                                   target_mels=target_mels)
        return decoder_output, attn_logits


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
