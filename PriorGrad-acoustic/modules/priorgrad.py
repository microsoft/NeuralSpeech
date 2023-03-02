# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from modules.operations import *
from modules.tts_modules import DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor,\
    TransformerEncoderLayer, DEFAULT_MAX_SOURCE_POSITIONS
from modules.diffusion import DiffDecoder
from tts_utils.world_utils import f0_to_coarse_torch, restore_pitch
import numpy as np


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


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


class PriorGrad(nn.Module):
    def __init__(self, arch, dictionary, out_dims=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        if isinstance(arch, str):
            self.arch = list(map(int, arch.strip().split()))
        else:
            assert isinstance(arch, (list, tuple))
            self.arch = arch
        self.enc_layers = hparams['enc_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = nn.Embedding(len(self.dictionary), self.hidden_size, self.padding_idx)
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)

        self.decoder_proj_dim = hparams['decoder_proj_dim'] if 'decoder_proj_dim' in hparams else self.hidden_size
        if self.decoder_proj_dim != self.hidden_size:
            self.decoder_proj = Linear(self.hidden_size, hparams['decoder_proj_dim'], bias=True)


        self.use_phone_stat = hparams['use_phone_stat'] if 'use_phone_stat' in hparams else False
        self.condition_phone_stat = hparams['condition_phone_stat'] if 'condition_phone_stat' in hparams else False

        self.diff_params = AttrDict(
            n_mels = hparams['audio_num_mel_bins'],
            residual_layers = hparams['diff_residual_layers'],
            residual_channels = hparams['diff_residual_channels'],
            conditioner_channels = self.decoder_proj_dim,
            dilation_cycle_length = 1,
            noise_schedule = np.linspace(hparams['diff_beta_start'], hparams['diff_beta_end'], hparams['diff_num_steps']).tolist(),
            inference_noise_schedule = hparams['diff_inference_noise_schedule'],
            use_phone_stat = self.use_phone_stat,
            condition_phone_stat = self.condition_phone_stat

        )

        self.diff_beta = np.array(self.diff_params.noise_schedule)
        noise_level = np.cumprod(1 - self.diff_beta)
        self.diff_noise_level = torch.tensor(noise_level.astype(np.float32))

        self.decoder = DiffDecoder(self.diff_params)

        if hparams['use_spk_id']:
            self.spk_embed_proj = nn.Embedding(hparams['num_spk'], self.hidden_size)
        else:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            n_chans=hparams['predictor_hidden'],
            dropout_rate=0.5, padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = nn.Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5,
                padding=hparams['ffn_padding'], odim=2)
            self.pitch_do = nn.Dropout(0.5)
        if hparams['use_energy_embed']:
            self.energy_predictor = EnergyPredictor(
                self.hidden_size, n_chans=hparams['predictor_hidden'], dropout_rate=0.5, odim=1,
                padding=hparams['ffn_padding'])
            self.energy_embed = nn.Embedding(256, self.hidden_size, self.padding_idx)
            self.energy_do = nn.Dropout(0.5)

        # encoder proj for MAS
        if hparams['dur'] == 'mas':
            print("INFO: using monotonic alignment search (MAS) for duration predictor training!")
            self.encoder_proj = Linear(self.hidden_size, hparams['audio_num_mel_bins'], bias=True)

    def forward(self, src_tokens, mel2ph, spk_embed=None,
                ref_mels=None, target_mean=None, target_std=None, target_nonpadding=None, pitch=None, uv=None, energy=None,
                is_training=True, fast_sampling=False, skip_decoder=False):
        """

        :param src_tokens: [B, T]
        :param mel2ph:
        :param spk_embed:
        :param ref_mels:
        :param target_nonpadding: pre-computed mask for ref_mels
        :return: {
            'mel_out': [B, T_s, 80], 'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        B, T_text = src_tokens.shape

        ret = {}
        encoder_outputs = self.encoder(src_tokens)
        encoder_out = encoder_outputs['encoder_out']  # [T, B, C]

        src_nonpadding = (src_tokens > 0).float().permute(1, 0)[:, :, None]
        if hparams['use_spk_embed'] and spk_embed is not None:
            spk_embed = self.spk_embed_proj(spk_embed)[None, :, :]
            encoder_out += spk_embed
        encoder_out = encoder_out * src_nonpadding  # [T, B, C]

        dur_input = encoder_out.transpose(0, 1)
        if hparams['predictor_sg']:
            dur_input = dur_input.detach()

        if hparams['dur'] == 'mfa': # original FS2 with GT mel2ph
            if mel2ph is None:
                dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
                ret['phoneme_aligned'] = torch.repeat_interleave(src_tokens[0], dur[0]).unsqueeze(0)
                if not hparams['sep_dur_loss']:
                    dur[src_tokens == self.dictionary.seg()] = 0
                ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
            else:
                ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)

        elif hparams['dur'] == 'mas': # modified FS2 with MAS, and without access to GT duration
            import monotonic_align
            if mel2ph is None:
                dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
                ret['phoneme_aligned'] = torch.repeat_interleave(src_tokens[0], dur[0]).unsqueeze(0)
                if not hparams['sep_dur_loss']:
                    dur[src_tokens == self.dictionary.seg()] = 0
                ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
                encoder_proj = self.encoder_proj(encoder_out)
                encoder_proj_aligned = F.pad(encoder_proj, [0, 0, 0, 0, 1, 0])
                mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_proj_aligned.shape[-1]]).contiguous()
                encoder_proj_aligned = torch.gather(encoder_proj_aligned, 0, mel2ph_).transpose(0, 1)  # [B, T_mel, 80]
                ret['encoder_proj_aligned'] = encoder_proj_aligned # for POC inference

            else:
                # make sure that we don't mistakenly use MFA mel2ph
                mel2ph = None
                # MAS requires [B, T_text, T_mel] "score-like" tensor to calculate optimal path
                # build [B, T_text, T_mel] l2-distance tensor and run search
                encoder_proj = self.encoder_proj(encoder_out) * src_nonpadding # [T_text, B, 80] * [T_text, B, 1]

                # L2 distance btw [B, T_text, 80] and [B, T_mel, 80] with cdist
                distance_matrix_l2 = torch.cdist(encoder_proj.transpose(0, 1), ref_mels, p=2) # [B, T_text, T_mel]

                # search for optimal path btw encoder output & target mel using MAS, with negative distance as input
                with torch.no_grad():
                    src_nonpadding_mas = (src_tokens > 0).float()  # [B, T_text]
                    attn_mask = torch.unsqueeze(src_nonpadding_mas, -1) * torch.unsqueeze(target_nonpadding, 1) # [B, T_text, T_mel]
                    optimal_path = monotonic_align.maximum_path(-distance_matrix_l2, attn_mask) # [B, T_text, T_mel]
                    optimal_path = optimal_path.detach()
                    # get "GT" token duration from MAS: sum over T_mel dim to get linear duration & apply source mask
                    dur_gt_mas = torch.sum(optimal_path, -1).long() * src_nonpadding_mas.long()

                    # get mel2ph using dur_gt_mas with length regulator module, this overwrites MFA-given mel2ph
                    # duration loss will be computed with ret['dur'] and ret['mel2ph_mas']
                    ret['mel2ph_mas'] = mel2ph = self.length_regulator(dur_gt_mas, (src_tokens !=0).sum(-1))[..., 0]

                encoder_proj_aligned = torch.matmul(optimal_path.transpose(1, 2), encoder_proj.transpose(0, 1)) # [B, T_mel, 80]
                ret['encoder_proj_aligned'] = encoder_proj_aligned # to be used for encoder loss
                ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)

        # expand encoder out to make decoder inputs
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])
        mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()
        decoder_inp = torch.gather(decoder_inp, 0, mel2ph_).transpose(0, 1)  # [B, T, H]
        ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp

        # add pitch embed
        if hparams['use_pitch_embed']:
            decoder_inp = decoder_inp + self.add_pitch(decoder_inp_origin, pitch, uv, mel2ph, ret)
        # add energy embed
        if hparams['use_energy_embed']:
            decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin, energy, ret)

        if self.decoder_proj_dim != self.hidden_size:
            decoder_inp = self.decoder_proj(decoder_inp) # [T, B, proj_dim]

        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            return ret

        # run diffusion
        if self.use_phone_stat:
            assert target_mean is not None and target_std is not None, "use_phone_stat is true but mean and std are None"
            # expand target_mean and target_std accordingly
            mel2ph__ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, target_mean.shape[-1]]).contiguous()
            target_mean_ = F.pad(target_mean.permute(1, 0, 2), [0, 0, 0, 0, 1, 0])
            target_mean_ = torch.gather(target_mean_, 0, mel2ph__).transpose(0, 1)
            target_std_ = F.pad(target_std.permute(1, 0, 2), [0, 0, 0, 0, 1, 0], value=1.)
            target_std_ = torch.gather(target_std_, 0, mel2ph__).transpose(0, 1)
            target_mean, target_std = target_mean_, target_std_
            ret['target_mean_aligned'] = target_mean
            ret['target_std_aligned'] = target_std

        # use X-mu, zero-mean data for training, and add mean at the last step of diffusion
        if is_training: # compute diffusion loss
            t = torch.randint(0, len(self.diff_params.noise_schedule), [B])
            noise_scale = self.diff_noise_level[t].unsqueeze(-1).unsqueeze(-1).to(src_tokens.device)
            noise_scale_sqrt = noise_scale ** 0.5

            if self.use_phone_stat:
                noise = torch.randn_like(ref_mels) * target_std
                noisy_mel = noise_scale_sqrt * (ref_mels - target_mean) + (1.0 - noise_scale) ** 0.5 * noise  # use X - mu
            else:
                noise = torch.randn_like(ref_mels)
                noisy_mel = noise_scale_sqrt * ref_mels + (1.0 - noise_scale) ** 0.5 * noise

            noise_pred = self.decoder(noisy_mel, decoder_inp, target_mean, target_std, mel2ph, t) # we can still use mu and std as condition, this will predict N(0, Sigma)

            ret['noise_pred'] = noise_pred
            if target_mean is not None:
                ret['noise_target'] = (noise) * (mel2ph != 0).float()[:, :, None] # apply mask to noise for correct loss, noise is already
            else:
                ret['noise_target'] = noise * (mel2ph != 0).float()[:, :, None]

        else: # run reverse diffusion sampling
            x_mel, x_mel_list = self.decoder.sample(decoder_inp, target_mean, target_std, mel2ph, fast_sampling=fast_sampling, return_all=True)
            x_mel = x_mel * (mel2ph != 0).float()[:, :, None]
            ret['mel_out'] = x_mel

        return ret

    def decode_with_pred_pitch(self, decoder_inp, mel2ph):
        if hparams['use_ref_enc']:
            assert False
        pitch_embed = self.add_pitch(decoder_inp, None, None, mel2ph, {})
        decoder_inp = decoder_inp + self.pitch_do(pitch_embed)
        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        x = decoder_inp
        x = self.decoder(x)
        x = self.mel_out(x)
        x = x * (mel2ph != 0).float()[:, :, None]
        return x

    # run other modules
    def add_energy(self, decoder_inp, energy, ret):
        if hparams['predictor_sg']:
            decoder_inp = decoder_inp.detach()
        ret['energy_pred'] = energy_pred = self.energy_predictor(decoder_inp)[:, :, 0]
        if energy is None:
            energy = energy_pred
        energy = torch.clamp(torch.div(energy * 256, 4, rounding_mode='floor'), min = 0, max=255).long()
        energy_embed = self.energy_embed(energy)
        return energy_embed

    def add_pitch(self, decoder_inp_origin, pitch, uv, mel2ph, ret):
        pp_inp = decoder_inp_origin
        if hparams['predictor_sg']:
            pp_inp = pp_inp.detach()
        ret['pitch_logits'] = pitch_logits = self.pitch_predictor(pp_inp)
        if pitch is not None:  # train
            pitch_padding = pitch == -200
            pitch_restore = restore_pitch(pitch, uv if hparams['use_uv'] else None, hparams,
                                          pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        else:  # test
            pitch_padding = (mel2ph == 0)
            pitch = pitch_logits[:, :, 0]
            uv = pitch_logits[:, :, 1] > 0
            if not hparams['use_uv']:
                uv = pitch < -3.5
            pitch_restore = restore_pitch(pitch, uv, hparams, pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            ret['uv'] = uv
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            pitch_embed = self.pitch_embed(pitch_restore)
        return self.pitch_do(pitch_embed)
