# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from modules.operations import *
from modules.tts_modules import TransformerEncoder, LightSpeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor
import utils
from utils.world_utils import f0_to_coarse_torch, restore_pitch


class LightSpeech(nn.Module):
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
        self.dec_layers = hparams['dec_layers']
        self.enc_arch = self.arch[:self.enc_layers]
        self.dec_arch = self.arch[self.enc_layers:self.enc_layers + self.dec_layers]
        self.hidden_size = hparams['hidden_size']
        self.encoder_embed_tokens = nn.Embedding(len(self.dictionary), self.hidden_size, self.padding_idx)
        self.encoder = TransformerEncoder(self.enc_arch, self.encoder_embed_tokens)
        self.decoder = LightSpeechDecoder(self.dec_arch) if hparams['dec_layers'] > 0 else None
        self.mel_out = Linear(self.hidden_size,
                              hparams['audio_num_mel_bins'] if out_dims is None else out_dims,
                              bias=True)
        if hparams['use_spk_id']:
            self.spk_embed_proj = nn.Embedding(hparams['num_spk'], self.hidden_size)
        else:
            self.spk_embed_proj = Linear(256, self.hidden_size, bias=True)
        self.dur_predictor = DurationPredictor(
            self.hidden_size,
            hparams['dur_predictor_layer'],
            n_chans=hparams['predictor_hidden'],
            dropout_rate=0.5, padding=hparams['ffn_padding'],
            kernel_size=hparams['dur_predictor_kernel'])
        self.length_regulator = LengthRegulator()
        if hparams['use_pitch_embed']:
            self.pitch_embed = nn.Embedding(300, self.hidden_size, self.padding_idx)
            self.pitch_predictor = PitchPredictor(
                self.hidden_size, hparams['pitch_predictor_layer'], n_chans=hparams['predictor_hidden'], dropout_rate=0.5,
                padding=hparams['ffn_padding'], odim=2)
            self.pitch_do = nn.Dropout(0.5)
        if hparams['use_energy_embed']:
            self.energy_predictor = EnergyPredictor(
                self.hidden_size, hparams['energy_predictor_layer'], n_chans=hparams['predictor_hidden'], dropout_rate=0.5, odim=1,
                padding=hparams['ffn_padding'])
            self.energy_embed = nn.Embedding(256, self.hidden_size, self.padding_idx)
            self.energy_do = nn.Dropout(0.5)

    def forward(self, src_tokens, mel2ph, spk_embed=None,
                ref_mels=None, pitch=None, uv=None, energy=None, skip_decoder=False):
        """

        :param src_tokens: [B, T]
        :param mel2ph:
        :param spk_embed:
        :param ref_mels:
        :return: {
            'mel_out': [B, T_s, 80], 'dur': [B, T_t],
            'w_st_pred': [heads, B, tokens], 'w_st': [heads, B, tokens],
            'encoder_out_noref': [B, T_t, H]
        }
        """
        ret = {}
        #with utils.Timer('encoder', print_time=hparams['profile_infer']):
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
        #with utils.Timer('dur', print_time=hparams['profile_infer']):
        if mel2ph is None:
            dur = self.dur_predictor.inference(dur_input, src_tokens == 0)
            if not hparams['sep_dur_loss']:
                dur[src_tokens == self.dictionary.seg()] = 0
            ret['mel2ph'] = mel2ph = self.length_regulator(dur, (src_tokens != 0).sum(-1))[..., 0]
        else:
            ret['dur'] = self.dur_predictor(dur_input, src_tokens == 0)

        # expand encoder out to make decoder inputs
        decoder_inp = F.pad(encoder_out, [0, 0, 0, 0, 1, 0])
        mel2ph_ = mel2ph.permute([1, 0])[..., None].repeat([1, 1, encoder_out.shape[-1]]).contiguous()
        decoder_inp = torch.gather(decoder_inp, 0, mel2ph_).transpose(0, 1)  # [B, T, H]
        ret['decoder_inp_origin'] = decoder_inp_origin = decoder_inp

        # add pitch embed
        if hparams['use_pitch_embed']:
            #with utils.Timer('pitch', print_time=hparams['profile_infer']):
            decoder_inp = decoder_inp + self.add_pitch(decoder_inp_origin, pitch, uv, mel2ph, ret)
        # add energy embed
        if hparams['use_energy_embed']:
            with utils.Timer('energy', print_time=hparams['profile_infer']):
                decoder_inp = decoder_inp + self.add_energy(decoder_inp_origin, energy, ret)

        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
            #with utils.Timer('decoder', print_time=hparams['profile_infer']):
            x = self.decoder(x)
        x = self.mel_out(x)
        x = x * (mel2ph != 0).float()[:, :, None]
        ret['mel_out'] = x
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
        energy = torch.clamp(energy * 256 // 4, max=255).long()
        energy_embed = self.energy_embed(energy)
        return energy_embed

    def add_pitch(self, decoder_inp_origin, pitch, uv, mel2ph, ret):
        pp_inp = decoder_inp_origin
        if hparams['predictor_sg']:
            pp_inp = pp_inp.detach()
        #with utils.Timer('pitch_predictor', print_time=hparams['profile_infer']):
        ret['pitch_logits'] = pitch_logits = self.pitch_predictor(pp_inp)
        if pitch is not None:  # train
            pitch_padding = pitch == -200
            #with utils.Timer('restore_pitch', print_time=hparams['profile_infer']):
            pitch_restore = restore_pitch(pitch, uv if hparams['use_uv'] else None, hparams,
                                          pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            #with utils.Timer('f0_to_coarse', print_time=hparams['profile_infer']):
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            #with utils.Timer('pitch_embed', print_time=hparams['profile_infer']):
            pitch_embed = self.pitch_embed(pitch_restore)
        else:  # test
            pitch_padding = (mel2ph == 0)
            pitch = pitch_logits[:, :, 0]
            uv = pitch_logits[:, :, 1] > 0
            if not hparams['use_uv']:
                uv = pitch < -3.5
            #with utils.Timer('restore_pitch', print_time=hparams['profile_infer']):
            pitch_restore = restore_pitch(pitch, uv, hparams, pitch_padding=pitch_padding)
            ret['pitch'] = pitch_restore
            #with utils.Timer('f0_to_coarse', print_time=hparams['profile_infer']):
            pitch_restore = f0_to_coarse_torch(pitch_restore)
            #with utils.Timer('pitch_embed', print_time=hparams['profile_infer']):
            pitch_embed = self.pitch_embed(pitch_restore)
        return self.pitch_do(pitch_embed)
