# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import re
import glob
import logging
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from tqdm import tqdm

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.distributed as dist

from modules.lightspeech import LightSpeech
from modules.tts_modules import DurationPredictorLoss
from tasks.base_task import BaseDataset, BaseTask
import utils
from utils.pl_utils import data_loader
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.text_encoder import TokenTextEncoder
from utils import audio
from utils.pwg_decode_from_mel import generate_wavegan, load_pwg_model
from utils.plot import plot_to_figure
from utils.world_utils import restore_pitch, process_f0 
from utils.tts_utils import GeneralDenoiser
from g2p_en import G2p

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

class LightSpeechDataset(BaseDataset):
    """A dataset that provides helpers for batching."""

    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False, infer_only=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.infer_only = infer_only
        if not self.infer_only:
            self.data = None
            self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.num_spk = hparams['num_spk']
        self.use_indexed_ds = hparams['indexed_ds']
        self.indexed_bs = None
        self.g2p = G2p()

        if not self.infer_only:
            # filter out items with no pitch
            f0s = np.load(f'{self.data_dir}/{prefix}_f0s.npy', allow_pickle=True)
            self.avail_idxs = [i for i, f0 in enumerate(f0s) if sum(f0) > 0]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        # pitch stats
        f0s = np.load(f'{self.data_dir}/train_f0s.npy', allow_pickle=True)
        f0s = np.concatenate(f0s, 0)
        f0s = f0s[f0s != 0]
        hparams['f0_mean'] = self.f0_mean = np.mean(f0s).item()
        hparams['f0_std'] = self.f0_std = np.std(f0s).item()

    def text_to_phone(self, txt):
        # function that converts the user-given text to phoneme sequence used in LightSpeech
        # the implementation mirrors datasets/tts/lj/prepare.py and datasets/tts/lj/gen_fs2_p.py
        # input: text string
        # output: encoded phoneme string
        phs = [p.replace(" ", "|") for p in self.g2p(txt)]
        ph = " ".join(phs)
        ph = "<UNK> " + ph + " <EOS>"
        phone_encoded = self.phone_encoder.encode(ph)
        return phone_encoded

    def _get_item(self, index):
        if not self.use_indexed_ds:
            key = self.idx2key[index]
            item = np.load(f'{self.data_dir}/{self.prefix}/{key}.npy', allow_pickle=True).item()
        else:
            if self.indexed_bs is None:
                self.indexed_bs = IndexedDataset(f'{self.data_dir}/{self.prefix}')
            item = self.indexed_bs[index]
        return item

    def __getitem__(self, index):
        hparams = self.hparams
        index = self.avail_idxs[index]
        key = self.idx2key[index]
        item = self._get_item(index)
        spec = torch.Tensor(item['mel'])
        energy = (spec.exp() ** 2).sum(-1).sqrt()[:hparams['max_frames']]
        mel2ph = torch.LongTensor(item['mel2ph'])[:hparams['max_frames']]
        f0, uv = process_f0(item["f0"], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])

        sample = {
            "id": index,
            "utt_id": key,
            "text": item['txt'],
            "source": phone,
            "target": spec[:hparams['max_frames']],
            "pitch": torch.LongTensor(item.get("pitch"))[:hparams['max_frames']],
            "energy": energy,
            "f0": f0[:hparams['max_frames']],
            "uv": uv[:hparams['max_frames']],
            "mel2ph": mel2ph,
        }
        if self.num_spk > 1:
            sample["spk_id"] = item['spk_id']
            sample["spk_embed"] = item['spk_embed']
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        pad_idx = self.phone_encoder.pad()
        id = torch.LongTensor([s['id'] for s in samples])
        utt_ids = [s['utt_id'] for s in samples]
        text = [s['text'] for s in samples]

        src_tokens = utils.collate_1d([s['source'] for s in samples], pad_idx)
        f0 = utils.collate_1d([s['f0'] for s in samples], -200) if self.hparams['use_pitch_embed'] else None
        uv = utils.collate_1d([s['uv'] for s in samples]) if self.hparams['use_pitch_embed'] else None
        energy = utils.collate_1d([s['energy'] for s in samples], pad_idx) if self.hparams['use_energy_embed'] else None
        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], pad_idx)
        target = utils.collate_2d([s['target'] for s in samples], pad_idx)
        prev_output_mels = utils.collate_2d([s['target'] for s in samples], pad_idx, shift_right=True)

        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        target_lengths = torch.LongTensor([s['target'].shape[0] for s in samples])
        ntokens = sum(len(s['source']) for s in samples)
        nmels = sum(len(s['target']) for s in samples)

        batch = {
            'id': id,
            'utt_id': utt_ids,
            'nsamples': len(samples),
            'ntokens': ntokens,
            'nmels': nmels,
            'text': text,
            'src_tokens': src_tokens,
            'mel2ph': mel2ph,
            'src_lengths': src_lengths,
            'targets': target,
            'energy': energy,
            'target_lengths': target_lengths,
            'prev_output_mels': prev_output_mels,
            'pitch': f0,
            'uv': uv,
        }

        if self.num_spk > 1:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            spk_embed = torch.FloatTensor([s['spk_embed'] for s in samples])
            batch['spk_ids'] = spk_ids
            batch['spk_embed'] = spk_embed
        return batch


class RSQRTSchedule(object):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.constant_lr = hparams['lr']
        self.warmup_updates = hparams['warmup_updates']
        self.hidden_size = hparams['hidden_size']
        self.lr = hparams['lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr
        self.step(0)

    def step(self, num_updates):
        constant_lr = self.constant_lr
        warmup = min(num_updates / self.warmup_updates, 1.0)
        rsqrt_decay = max(self.warmup_updates, num_updates) ** -0.5
        rsqrt_hidden = self.hidden_size ** -0.5
        self.lr = max(constant_lr * warmup * rsqrt_decay * rsqrt_hidden, 1e-7)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class LightSpeechTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.arch = hparams['arch']
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = utils.get_num_heads(self.arch[hparams['enc_layers']:])
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}      
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()

    @data_loader
    def train_dataloader(self):
        train_dataset = LightSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = LightSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = LightSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False):
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1

        if max_tokens is not None:
            max_tokens *= torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
        if max_sentences is not None:
            max_sentences *= torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
        indices = dataset.ordered_indices()
        batch_sampler = utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        if shuffle:
            batches = torch.utils.data.sampler.SubsetRandomSampler(batch_sampler)
            if self.trainer.use_ddp:
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
                batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
                batches = torch.utils.data.sampler.SubsetRandomSampler(batches)
        else:
            batches = batch_sampler
            if self.trainer.use_ddp:
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
                batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(
                    dataset,
                    collate_fn=dataset.collater,
                    batch_sampler=batches,
                    num_workers=dataset.num_workers,
                    pin_memory=True
                )

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list)

    def build_model(self):
        arch = self.arch
        model = LightSpeech(arch, self.phone_encoder)
        total_params = utils.count_parameters(model)
        logging.info('Model Size: {}'.format(total_params))
        return model

    def build_scheduler(self, optimizer):
        return RSQRTSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def _training_step(self, sample, batch_idx, _):
        input = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        pitch = sample['pitch']
        energy = sample['energy']
        uv = sample['uv']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        loss_output, output = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                             pitch=pitch, uv=uv, energy=energy,
                                             return_output=True)
        total_loss = sum([v for v in loss_output.values() if v.requires_grad])
        loss_output['batch_size'] = target.size()[0]
        return total_loss, loss_output

    def loss(self, decoder_output, target):
        # decoder_output : B x T x (mel+1)
        # target : B x T x mel
        predicted_mel = decoder_output[:, :, :hparams['audio_num_mel_bins']]
        predicted_stop = decoder_output[:, :, -1]
        seq_mask, stop_mask = self.make_stop_target(target)

        l1_loss = F.l1_loss(predicted_mel, target, reduction='none')
        l2_loss = F.mse_loss(predicted_mel, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        l2_loss = (l2_loss * weights).sum() / weights.sum()
        stop_loss = (self.weighted_cross_entropy_with_logits(stop_mask, predicted_stop,
                                                             hparams['stop_token_weight']) * seq_mask).sum()
        stop_loss = stop_loss / (seq_mask.sum() + target.size(0) * (hparams['stop_token_weight'] - 1))

        return {
            'l1': l1_loss,
            'l2': l2_loss,
            'stop_loss': stop_loss,
        }

    def validation_step(self, sample, batch_idx):
        input = sample['src_tokens']
        target = sample['targets']
        mel2ph = sample['mel2ph']
        pitch = sample['pitch']
        energy = sample['energy']
        uv = sample['uv']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, input, mel2ph, spk_embed, target,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy,
                                                      return_output=True)
        outputs['total_loss'] = outputs['losses']['mel']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        if batch_idx < 10:
            if 'pitch_logits' in model_out:
                pitch[uv > 0] = -4
                pitch_pred = model_out['pitch_logits'][:, :, 0]
                pitch_pred[model_out['pitch_logits'][:, :, 1] > 0] = -4
                self.logger.experiment.add_figure(f'pitch_{batch_idx}', plot_to_figure({
                    'gt': pitch[0].detach().cpu().numpy(),
                    'pred': pitch_pred[0].detach().cpu().numpy()
                }), self.global_step)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AverageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AverageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def test_step(self, sample, batch_idx):
        logging.info('inferring batch {} with {} samples'.format(batch_idx, sample['nsamples']))
        with utils.Timer('trans_tts', print_time=hparams['profile_infer']):
            decoded_mel, encdec_attn, hit_eos, _, focus_rate, phone_coverage_rate, diagonal_focus_rate = \
                self.infer_batch(sample)

        hit_eos = hit_eos[:, 1:]
        outputs = decoded_mel
        predict_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
        outputs *= (1.0 - hit_eos.float())[:, :, None]

        sample['outputs'] = outputs
        sample['predict_mels'] = decoded_mel
        sample['predict_lengths'] = predict_lengths
        sample['encdec_attn'] = encdec_attn
        self.after_infer(sample)

    def infer_batch(self, sample):
        model = self.model
        input = sample['src_tokens']
        bsz = input.size(0)
        max_input_len = input.size(1)

        decode_length = self.estimate_decode_length(max_input_len)
        encoder_outputs = model.forward_encoder(input)
        encoder_out = encoder_outputs['encoder_out']
        encoder_padding_mask = encoder_outputs['encoder_padding_mask']

        hit_eos = input.new(bsz, 1).fill_(0).bool()
        stop_logits = input.new(bsz, 0).fill_(0).float()
        stage = 0
        decoder_input = input.new(bsz, decode_length + 1, hparams['audio_num_mel_bins']).fill_(
            0).float()
        decoded_mel = input.new(bsz, 0, hparams['audio_num_mel_bins']).fill_(0).float()
        encdec_attn_logits = []

        for i in range(hparams['dec_layers']):
            encdec_attn_logits.append(input.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
        attn_pos = input.new(bsz).fill_(0).int()
        use_masks = []
        for i in range(hparams['dec_layers']):
            use_masks.append(input.new(self.num_heads[i]).fill_(0).float())

        incremental_state = {}
        step = 0
        if hparams['attn_constraint']:
            for i, layer in enumerate(model.decoder.layers):
                enc_dec_attn_constraint_mask = input.new(bsz, self.num_heads[i], max_input_len).fill_(0).int()
                layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)

        def is_finished(step, decode_length, hit_eos, stage):
            finished = step >= decode_length
            finished |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
            if hparams['attn_constraint']:
                finished &= stage != 0
            return finished

        while True:
            if is_finished(step, decode_length, hit_eos, stage):
                break

            decoder_output, attn_logits = model.forward_decoder(
                decoder_input[:, :step + 1], encoder_out,
                encoder_padding_mask,
                incremental_state=incremental_state
            )
            next_mel = decoder_output[:, -1:, :hparams['audio_num_mel_bins']]
            stop_logit = decoder_output[:, -1:, -1]
            stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
            decoded_mel = torch.cat((decoded_mel, next_mel), dim=1)
            for i in range(hparams['dec_layers']):
                encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
            step += 1

            this_hit_eos = hit_eos[:, -1:]
            if hparams['attn_constraint']:
                this_hit_eos |= (attn_pos[:, None] >= (encoder_padding_mask < 1.0).float().sum(dim=-1,
                                        keepdim=True).int() - 5) & (torch.sigmoid(stop_logit) > 0.5)
            else:
                this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
            hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)

            decoder_input[:, step] = next_mel[:, -1]

            if hparams['attn_constraint']:
                stage_change_step = 50
                all_prev_weights = []
                for i in range(hparams['dec_layers']):
                    all_prev_weights.append(torch.softmax(encdec_attn_logits[i], dim=-1))  # bsz x head x L x L_kv

                # if the stage should change
                next_stage = (step == stage_change_step) | (step >= decode_length)
                next_stage |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
                next_stage &= (stage == 0)

                # choose the diagonal attention
                if next_stage:  # TODO
                    use_masks = []
                    for i in range(hparams['dec_layers']):
                        use_mask = (all_prev_weights[i][:, :, :step].max(dim=-1).values.mean(
                            dim=(0, 2)) > 0.6).float()  # [head]
                        use_masks.append(use_mask)
                    attn_pos = input.new(bsz).fill_(0).int()

                    # reseet when the stage changes
                    for layer in model.decoder.layers:
                        layer.clear_buffer(input, encoder_out, encoder_padding_mask, incremental_state)

                    encdec_attn_logits = []
                    for i in range(hparams['dec_layers']):
                        encdec_attn_logits.append(
                            input.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
                    decoded_mel = input.new(bsz, 0, hparams['audio_num_mel_bins']).fill_(0).float()
                    decoder_input = input.new(bsz, decode_length + 1, hparams['audio_num_mel_bins']).fill_(
                        0).float()
                    hit_eos = input.new(bsz, 1).fill_(0).bool()
                    stage = stage + 1
                    step = 0

                prev_weights_mask1 = utils.sequence_mask(
                    torch.max(attn_pos - 1, attn_pos.new(attn_pos.size()).fill_(0)).float(),
                    encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                prev_weights_mask2 = 1.0 - utils.sequence_mask(attn_pos.float() + 4,
                                                               encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                enc_dec_attn_constraint_masks = []
                for i in range(hparams['dec_layers']):
                    mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[i][None, :, None]  # bsz x head x L_kv
                    enc_dec_attn_constraint_masks.append(mask)
                
                for i, layer in enumerate(model.decoder.layers):
                    enc_dec_attn_constraint_mask = enc_dec_attn_constraint_masks[i]
                    layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask,
                                     incremental_state)

                def should_move_on():
                    prev_weights = []
                    for i in range(hparams['dec_layers']):
                        prev_weight = (all_prev_weights[i] * use_masks[i][None, :, None, None]).sum(dim=1)
                        prev_weights.append(prev_weight)
                    prev_weights = sum(prev_weights) / sum([mask.sum() for mask in use_masks])
                    move_on = (prev_weights[:, -3:].mean(dim=1).gather(1, attn_pos[:, None].long())).squeeze(-1) < 0.7
                    move_on &= torch.argmax(prev_weights[:, -1], -1) > attn_pos.long()
                    return attn_pos + move_on.int()

                if step > 3 and stage == 1:
                    attn_pos = should_move_on()

        encdec_attn = utils.select_attn(encdec_attn_logits)

        if not hparams['profile_infer']:
            src_lengths = sample['src_lengths'] - 1  # exclude eos
            target_lengths = (1.0 - hit_eos[:, 1:].float()).sum(dim=-1) + 1
            src_padding_mask = input.eq(0) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            target_padding_mask = decoded_mel.abs().sum(-1).eq(0)
            focus_rate = utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask,
                                                                target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate, diag_mask = utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths,
                                                                           src_padding_mask,
                                                                           target_padding_mask)
        else:
            focus_rate, phone_coverage_rate, diagonal_focus_rate = None, None, None
        return decoded_mel, encdec_attn.unsqueeze(
            1), hit_eos, stop_logits, focus_rate, phone_coverage_rate, diagonal_focus_rate

    def estimate_decode_length(self, input_length):
        return input_length * 5 + 100

    def prepare_vocoder(self):
        if self.vocoder is None:
            if hparams['vocoder'] == 'pwg':
                if hparams['vocoder_ckpt'] == '':
                    base_dir = 'wavegan_pretrained'
                    ckpts = glob.glob(f'{base_dir}/checkpoint-*steps.pkl')
                    ckpt = sorted(ckpts, key=
                    lambda x: int(re.findall(f'{base_dir}/checkpoint-(\d+)steps.pkl', x)[0]))[-1]
                    config_path = f'{base_dir}/config.yml'
                else:
                    base_dir = hparams['vocoder_ckpt']
                    config_path = f'{base_dir}/config.yml'
                    ckpt = sorted(glob.glob(f'{base_dir}/model_ckpt_steps_*.ckpt'), key=
                    lambda x: int(re.findall(f'{base_dir}/model_ckpt_steps_(\d+).ckpt', x)[0]))[-1]
                print('| load wavegan: ', ckpt)
                self.vocoder = load_pwg_model(
                    config_path=config_path,
                    checkpoint_path=ckpt,
                    stats_path=f'{base_dir}/stats.h5',
                )
                self.denoiser = GeneralDenoiser()

    def inv_spec(self, spec, pitch=None, noise_spec=None):
        """

        :param spec: [T, 80]
        :return:
        """
        if hparams['vocoder'] == 'pwg':
            wav_out = generate_wavegan(spec, *self.vocoder, profile=hparams['profile_infer'])
            if hparams['gen_wav_denoise']:
                noise_out = generate_wavegan(noise_spec, *self.vocoder)[None, :] \
                    if noise_spec is not None else None
                wav_out = self.denoiser(wav_out[None, :], noise_out)[0, 0]
            wav_out = wav_out.cpu().numpy()
            return wav_out
    
    @staticmethod
    def save_result(wav_out, mel, prefix, utt_id, text, gen_dir,
                    pitch=None, noise_spec=None, alignment=None, str_phs=None):
        base_fn = f'[{prefix}][{utt_id}]'
        base_fn += text.replace(":", "%3A")[:80]
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        audio.plot_spec(mel.T, f'{gen_dir}/spec_plot/{base_fn}.png')
        if pitch is not None:
            audio.plot_curve(pitch, f'{gen_dir}/pitch_plot/{base_fn}.png', 50, 500)

        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower',
                           interpolation='none')
            decoded_txt = str_phs.split(" ")
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=6)
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close()

    def test_end(self, outputs):
        if self.saving_result_pool is not None:
            self.saving_result_pool.close()
            [f.get() for f in tqdm(self.saving_results_futures)]
            self.saving_result_pool.join()
        return {}

    def run_model(self, model, input, mel2ph, spk_embed, target,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None):
        hparams['global_steps'] = self.global_step
        losses = {}
        if ref_mel == 'tgt':
            ref_mel = target
        output = model(input, mel2ph, spk_embed, ref_mel, pitch, uv, energy)

        if hparams['mel_loss'] == 'l1':
            losses['mel'] = self.l1_loss(output['mel_out'], target)
        if hparams['mel_loss'] == 'mse':
            losses['mel'] = self.mse_loss(output['mel_out'], target)

        losses['dur'] = self.dur_loss(output['dur'], mel2ph, input)
        if hparams['use_pitch_embed']:
            p_pred = output['pitch_logits']
            losses['uv'], losses['f0'] = self.pitch_loss(p_pred, pitch, uv)
            if losses['uv'] is None:
                del losses['uv']

        if hparams['use_energy_embed']:
            losses['energy'] = self.energy_loss(output['energy_pred'], energy)

        if not return_output:
            return losses
        else:
            return losses, output

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def mse_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def dur_loss(self, dur_pred, mel2ph, input, split_pause=False, sent_dur_loss=False):
        B, T_t = input.shape
        dur_gt = mel2ph.new_zeros(B, T_t + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
        dur_gt = dur_gt[:, 1:]
        nonpadding = (input != 0).float()
        if split_pause:
            is_pause = (input == self.phone_encoder.seg()) | (input == self.phone_encoder.unk()) | (
                    input == self.phone_encoder.eos())
            is_pause = is_pause.float()
            phone_loss = self.dur_loss_fn(dur_pred, dur_gt, (1 - is_pause) * nonpadding) \
                         * hparams['lambda_dur']
            seg_loss = self.dur_loss_fn(dur_pred, dur_gt, is_pause) \
                       * hparams['lambda_dur']
            return phone_loss, seg_loss
        ph_dur_loss = self.dur_loss_fn(dur_pred, dur_gt, nonpadding) * hparams['lambda_dur']
        if not sent_dur_loss:
            return ph_dur_loss
        else:
            dur_pred = (dur_pred.exp() - 1).clamp(min=0) * nonpadding
            dur_gt = dur_gt.float() * nonpadding
            sent_dur_loss = F.l1_loss(dur_pred.sum(-1), dur_gt.sum(-1), reduction='none') / dur_gt.sum(-1)
            sent_dur_loss = sent_dur_loss.mean()
            return ph_dur_loss, sent_dur_loss

    def pitch_loss(self, p_pred, pitch, uv):
        assert p_pred[..., 0].shape == pitch.shape
        assert p_pred[..., 0].shape == uv.shape
        nonpadding = (pitch != -200).float().reshape(-1)
        if hparams['use_uv']:
            uv_loss = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1].reshape(-1), uv.reshape(-1), reduction='none') * nonpadding).sum() \
                      / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = (pitch != -200).float() * (uv == 0).float()
            nonpadding = nonpadding.reshape(-1)
        else:
            pitch[uv > 0] = -4
            uv_loss = None

        pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
        pitch_loss = (pitch_loss_fn(
            p_pred[:, :, 0].reshape(-1), pitch.reshape(-1), reduction='none') * nonpadding).sum() \
                     / nonpadding.sum() * hparams['lambda_pitch']
        return uv_loss, pitch_loss

    def energy_loss(self, energy_pred, energy):
        nonpadding = (energy != 0).float()
        loss = (F.mse_loss(energy_pred, energy, reduction='none') * nonpadding).sum() / nonpadding.sum()
        loss = loss * hparams['lambda_energy']
        return loss

    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        input = sample['src_tokens']
        if hparams['profile_infer']:
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            mel2ph = sample['mel2ph']
            pitch = sample['pitch']
            uv = sample['uv']
        else:
            mel2ph = None
            pitch = None
            uv = None
        with utils.Timer('model_time', print_time=hparams['profile_infer']):
            outputs = self.model(input, mel2ph, spk_embed, None, pitch, uv)

        # denoise
        if hparams['gen_wav_denoise']:
            mel2ph_pred = outputs['mel2ph']
            input_noise = torch.ones_like(input[:, :1]).long() * 3
            mel2ph_noise = torch.ones_like(mel2ph_pred)
            mel2ph_noise = mel2ph_noise * (mel2ph_pred > 0).long()
            mel2ph_noise = mel2ph_noise[:, :40]
            pitch_noise = torch.zeros_like(mel2ph_pred).float()[:, :40]
            uv_noise = torch.ones_like(mel2ph_pred)[:, :40]
            noise_outputs = self.model(input_noise, mel2ph_noise, spk_embed, None, pitch_noise, uv_noise)
            sample['noise_outputs'] = noise_outputs['mel_out']

        sample['outputs'] = outputs['mel_out']
        sample['pitch_pred'] = outputs.get('pitch')
        sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
        #if hparams['profile_infer']:
        #    return {}
        return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(8)
            self.saving_results_futures = []
        self.prepare_vocoder()
        predictions = utils.unpack_dict_to_list(predictions)
        if hparams['show_progress_bar']:
            t = tqdm(predictions)
        else:
            t = predictions
        for i, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            targets = self.remove_padding(prediction.get("targets"))
            outputs = self.remove_padding(prediction["outputs"])
            noise_outputs = self.remove_padding(prediction.get("noise_outputs"))
            pitch_pred = self.remove_padding(prediction.get("pitch_pred"))
            pitch_gt = self.remove_padding(prediction.get("pitch"), -200)

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.inv_spec(outputs, pitch_pred, noise_outputs)
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
                os.makedirs(f'{gen_dir}/pitch_plot', exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, outputs, f'P', utt_id, text, gen_dir, [pitch_pred, pitch_gt], noise_outputs]))

                wav_gt = self.inv_spec(targets, pitch_gt, noise_outputs)
                if targets is not None:
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, targets, 'G', utt_id, text, gen_dir, pitch_gt, noise_outputs]))
                t.set_description(
                    f"Pred_shape: {outputs.shape}, gt_shape: {targets.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}

    def remove_padding(self, x, padding_idx=0):
        return utils.remove_padding(x, padding_idx)

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        # target : B x T x mel
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - utils.sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r

    def weighted_cross_entropy_with_logits(self, targets, logits, pos_weight=1):
        x = logits
        z = targets
        q = pos_weight
        l = 1 + (q - 1) * z
        return (1 - z) * x + l * (torch.log(1 + torch.exp(-x.abs())) + F.relu(-x))



if __name__ == '__main__':
    LightSpeechTask.start()
