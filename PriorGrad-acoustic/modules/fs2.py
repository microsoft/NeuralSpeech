from modules.operations import *
from modules.transformer_tts import TransformerEncoder
from modules.tts_modules import FastspeechDecoder, DurationPredictor, LengthRegulator, PitchPredictor, EnergyPredictor
from tts_utils.world_utils import f0_to_coarse_torch, restore_pitch


class FastSpeech2(nn.Module):
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
        self.decoder = FastspeechDecoder(self.dec_arch) if hparams['dec_layers'] > 0 else None
        self.mel_out = Linear(self.hidden_size,
                              hparams['audio_num_mel_bins'] if out_dims is None else out_dims,
                              bias=True)
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

        print("encoder params:{}".format(sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)))
        print("decoder params:{}".format(sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)))

        # new impl: encoder proj for MAS
        if hparams['dur'] == 'mas':
            print("INFO: using monotonic alignment search (MAS) for duration predictor training!")
            self.encoder_proj = Linear(self.hidden_size, hparams['audio_num_mel_bins'], bias=True)

    def forward(self, src_tokens, mel2ph, spk_embed=None,
                ref_mels=None, target_nonpadding=None, pitch=None, uv=None, energy=None, skip_decoder=False):
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

        decoder_inp = decoder_inp * (mel2ph != 0).float()[:, :, None]
        ret['decoder_inp'] = decoder_inp

        if skip_decoder:
            return ret
        x = decoder_inp
        if hparams['dec_layers'] > 0:
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
