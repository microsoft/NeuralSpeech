import matplotlib

matplotlib.use('Agg')
from tts_utils.pl_utils import data_loader
import os, sys
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from tts_utils.hparams import hparams
from tts_utils.indexed_datasets import IndexedDataset
from tts_utils.plot import plot_to_figure, numpy_to_figure, spec_numpy_to_figure
from tts_utils.world_utils import restore_pitch, process_f0

import numpy as np

from modules.fs2 import FastSpeech2
from tasks.transformer_tts import TransformerTtsTask
from tasks.base_task import BaseDataset

import time

import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import tts_utils

sys.path.append("hifi-gan")


class FastSpeechDataset(BaseDataset):
    """A dataset that provides helpers for batching."""
    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.num_spk = hparams['num_spk']
        self.use_indexed_ds = hparams['indexed_ds']
        self.indexed_bs = None

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

        # phoneme stats loading
        self.use_phone_stat = hparams['use_phone_stat'] if 'use_phone_stat' in hparams else False
        if self.use_phone_stat:
            # loads phoneme statistics. all datasets use training stats
            print("INFO: using phoneme-level stats for PriorGrad modeling!")
            self.phone_to_mean = torch.from_numpy(np.load(f'{self.data_dir}/train_phone_to_mean.npy', allow_pickle=True))
            if hparams['use_std_norm']:
                print("INFO: using 0~1 normalized stds!")
                self.phone_to_std = torch.from_numpy(np.load(f'{self.data_dir}/train_phone_to_std_norm.npy', allow_pickle=True))
            else:
                print("INFO: using non-normalized stds!")
                self.phone_to_std = torch.from_numpy(np.load(f'{self.data_dir}/train_phone_to_std.npy', allow_pickle=True))
            print("INFO: phoneme mean stats: min {:.4f} max {:.4f} mean {:.4f} std {:.4f}".
                  format(self.phone_to_mean.min(), self.phone_to_mean.max(), self.phone_to_mean.mean(), self.phone_to_mean.std()))
            print("INFO: phoneme std stats: min {:.4f} max {:.4f} mean {:.4f} std {:.4f}".
                  format(self.phone_to_std.min(), self.phone_to_std.max(), self.phone_to_std.mean(), self.phone_to_std.std()))
            self.std_min = hparams['std_min']
            print("INFO: minimum of std is set to {}".format(self.std_min))
            self.std_max = hparams['std_max'] if 'std_max' in hparams else -1
            if self.std_max != -1:
                print("INFO: maximum of std is set to {}".format(self.std_max))
            self.use_std_only = hparams['use_std_only'] if 'use_std_only' in hparams else False
            if self.use_std_only:
                print("WARNING: use_std_only is true. phone_to_mean is wiped to all zero, falling back to N(0, sigma)!")
                self.phone_to_mean = torch.zeros_like(self.phone_to_mean)
                print("INFO: phoneme mean stats: min {:.4f} max {:.4f} mean {:.4f} std {:.4f}".
                      format(self.phone_to_mean.min(), self.phone_to_mean.max(), self.phone_to_mean.mean(), self.phone_to_mean.std()))
            self.use_mean_only = hparams['use_mean_only'] if 'use_mean_only' in hparams else False
            if self.use_mean_only:
                print("WARNING: use_mean_only is true. phone_to_std is wiped to all one, falling back to N(mu, I)!")
                self.phone_to_std = torch.ones_like(self.phone_to_std)
                print("INFO: phoneme std stats: min {:.4f} max {:.4f} mean {:.4f} std {:.4f}".
                      format(self.phone_to_std.min(), self.phone_to_std.max(), self.phone_to_std.mean(), self.phone_to_std.std()))

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

        if self.use_phone_stat:
            spec_mean = torch.index_select(self.phone_to_mean, 0, phone)
            spec_std = torch.index_select(self.phone_to_std, 0, phone)
            sample["target_mean"] = spec_mean
            sample["target_std"] = spec_std

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

        src_tokens = tts_utils.collate_1d([s['source'] for s in samples], pad_idx)
        f0 = tts_utils.collate_1d([s['f0'] for s in samples], -200) if self.hparams['use_pitch_embed'] else None
        uv = tts_utils.collate_1d([s['uv'] for s in samples]) if self.hparams['use_pitch_embed'] else None
        energy = tts_utils.collate_1d([s['energy'] for s in samples], pad_idx) if self.hparams['use_energy_embed'] else None
        mel2ph = tts_utils.collate_1d([s['mel2ph'] for s in samples], pad_idx)
        target = tts_utils.collate_2d([s['target'] for s in samples], pad_idx)
        prev_output_mels = tts_utils.collate_2d([s['target'] for s in samples], pad_idx, shift_right=True)

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

        if self.use_phone_stat:
            target_mean = tts_utils.collate_2d([s['target_mean'] for s in samples], pad_idx)
            target_std = tts_utils.collate_2d([s['target_std'] for s in samples], pad_idx)
            batch['targets_mean'] = target_mean
            # fill one instead of zero for target_std: zero value will cause NaN for scaled_mse_loss
            target_std[target_std == 0] = 1
            target_std[target_std <= self.std_min] = self.std_min
            if self.std_max != -1:
                target_std[target_std >= self.std_max] = self.std_max
            batch['targets_std'] = target_std

        if self.num_spk > 1:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            spk_embed = torch.FloatTensor([s['spk_embed'] for s in samples])
            batch['spk_ids'] = spk_ids
            batch['spk_embed'] = spk_embed
        return batch


class FastSpeech2Task(TransformerTtsTask):
    def __init__(self):
        super(FastSpeech2Task, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()

    @data_loader
    def train_dataloader(self):
        train_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['train_set_name'], hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                          hparams['valid_set_name'], hparams,
                                          shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = FastSpeechDataset(hparams['data_dir'], self.phone_encoder,
                                         hparams['test_set_name'], hparams, shuffle=False)
        return self.build_dataloader(test_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    def build_model(self):
        arch = self.arch
        model = FastSpeech2(arch, self.phone_encoder)
        return model

    def _training_step(self, sample, batch_idx, _):
        input = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        pitch = sample['pitch']
        energy = sample['energy']
        uv = sample['uv']

        # get mask for target beforehand for MAS
        # only get the first dim (previously repeated along mel bin)
        target_nonpadding = self.weights_nonzero_speech(target)[:, :, 0] # [B, T_mel]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        loss_output, output = self.run_model(self.model, input, mel2ph, spk_embed, target, target_nonpadding,
                                             pitch=pitch, uv=uv, energy=energy,
                                             return_output=True)
        total_loss = sum([v for v in loss_output.values() if v.requires_grad])
        loss_output['batch_size'] = target.size()[0]
        return total_loss, loss_output

    def validation_step(self, sample, batch_idx):
        input = sample['src_tokens']
        target = sample['targets']
        mel2ph = sample['mel2ph']
        pitch = sample['pitch']
        energy = sample['energy']
        uv = sample['uv']

        # get mask for target beforehand for MAS
        # only get the first dim (previously repeated along mel bin)
        target_nonpadding = self.weights_nonzero_speech(target)[:, :, 0] # [B, T_mel]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, input, mel2ph, spk_embed, target, target_nonpadding,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy,
                                                      return_output=True)
        outputs['total_loss'] = outputs['losses']['mel']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        outputs = tts_utils.tensors_to_scalars(outputs)
        if batch_idx < 10:
            if 'pitch_logits' in model_out:
                pitch[uv > 0] = -4
                pitch_pred = model_out['pitch_logits'][:, :, 0]
                pitch_pred[model_out['pitch_logits'][:, :, 1] > 0] = -4
                self.logger.experiment.add_figure(f'pitch_{batch_idx}', plot_to_figure({
                    'gt': pitch[0].detach().cpu().numpy(),
                    'pred': pitch_pred[0]
                        .detach().cpu().numpy()
                }), self.global_step)
            if 'mel_out' in model_out:
                mel_out = model_out['mel_out'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'mel_out_{batch_idx}', spec_numpy_to_figure(mel_out),
                                                  self.global_step)
            if 'encoder_proj_aligned' in model_out: # from MAS encoder_proj
                encoder_proj_aligned = model_out['encoder_proj_aligned'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'encoder_proj_aligned_{batch_idx}',spec_numpy_to_figure(encoder_proj_aligned),
                                                  self.global_step)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': tts_utils.AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = tts_utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    def run_model(self, model, input, mel2ph, spk_embed, target, target_nonpadding,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None):
        hparams['global_steps'] = self.global_step
        losses = {}
        if ref_mel == 'tgt':
            ref_mel = target

        output = model(input, mel2ph, spk_embed, ref_mel, target_nonpadding, pitch, uv, energy)

        if hparams['mel_loss'] == 'l1':
            losses['mel'] = self.l1_loss(output['mel_out'], target)
        if hparams['mel_loss'] == 'mse':
            losses['mel'] = self.mse_loss(output['mel_out'], target)

        if hparams['dur'] == 'mfa':
            losses['dur'] = self.dur_loss(output['dur'], mel2ph, input)
        elif hparams['dur'] == 'mas':
            assert 'mel2ph_mas' in output, "mel2ph_mas not found in model output!"
            assert 'encoder_proj_aligned' in output, "encoder_proj_aligned not found in model output!"
            losses['dur'] = self.dur_loss(output['dur'], output['mel2ph_mas'], input)
            if hparams['mel_loss'] == 'l1':
                losses['encoder'] = self.l1_loss(output['encoder_proj_aligned'], target)
            elif hparams['mel_loss'] == 'l2':
                losses['encoder'] = self.mse_loss(output['encoder_proj_aligned'], target)

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
        with tts_utils.Timer('fs', print_time=hparams['profile_infer']):
            tic = time.time()
            outputs = self.model(input, mel2ph, spk_embed, None, pitch, uv)
            torch.cuda.synchronize()
            toc  = time.time() - tic
            wav_dur = outputs['mel_out'].shape[1] * 256 / 22050.
            rtf = toc / wav_dur
            print("\nRTF: {:.4f}".format(rtf))

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
        sample['phoneme_aligned'] = outputs['phoneme_aligned']
        sample['uv'] = outputs['uv']
        if sample['pitch'] is not None:
            sample['pitch'] = restore_pitch(sample['pitch'], uv if hparams['use_uv'] else None, hparams)
        if 'encoder_proj_aligned' in outputs: # MAS only
            sample['encoder_proj_aligned'] = outputs['encoder_proj_aligned']

        return self.after_infer(sample)

    def prepare_vocoder_hfg(self):
        import json
        from env import AttrDict
        from models import Generator
        from inference import load_checkpoint
        checkpoint_file = "hifigan_pretrained/generator_v1"
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        global h
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        device = torch.device('cuda')
        self.vocoder = Generator(h).to(device)
        state_dict_g = load_checkpoint(checkpoint_file, device)
        self.vocoder.load_state_dict(state_dict_g['generator'])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

    def inv_spec_hfg(self, spec):
        """
        :param spec: [T, 80]
        :return:
        """
        spec = torch.FloatTensor(spec).unsqueeze(0).permute(0, 2, 1).cuda() # [B, 80 ,T]
        y_g_hat = self.vocoder(spec)
        audio = y_g_hat.squeeze().cpu().numpy()
        return audio

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(1)
            self.saving_results_futures = []
        if hparams['vocoder'] == 'hfg':
            self.prepare_vocoder_hfg()
        else:
            raise NotImplementedError("unknown vocoder")

        predictions = tts_utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            phoneme = prediction.get('phoneme_aligned')
            uv = prediction.get('uv')
            targets = self.remove_padding(prediction.get("targets"))
            outputs = self.remove_padding(prediction["outputs"])
            noise_outputs = self.remove_padding(prediction.get("noise_outputs"))
            pitch_pred = self.remove_padding(prediction.get("pitch_pred"))
            pitch_gt = self.remove_padding(prediction.get("pitch"), -200)

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            if hparams['vocoder'] == 'hfg':
                wav_pred = self.inv_spec_hfg(outputs)
            else:
                raise NotImplementedError("unknown vocoder")
            if not hparams['profile_infer']:
                os.makedirs(gen_dir, exist_ok=True)
                os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
                os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
                os.makedirs(f'{gen_dir}/pitch_plot', exist_ok=True)
                os.makedirs(f'{gen_dir}/spec', exist_ok=True)
                os.makedirs(f'{gen_dir}/text', exist_ok=True)
                os.makedirs(f'{gen_dir}/phoneme', exist_ok=True)
                os.makedirs(f'{gen_dir}/uv', exist_ok=True)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_pred, outputs, f'P', utt_id, text, gen_dir, [pitch_pred, pitch_gt], noise_outputs, None, None, phoneme, uv]))
                if hparams['vocoder'] == 'hfg':
                    wav_gt = self.inv_spec_hfg(targets)
                else:
                    raise NotImplementedError("unknown vocoder")

                if targets is not None:
                    self.saving_results_futures.append(
                        self.saving_result_pool.apply_async(self.save_result, args=[
                            wav_gt, targets, 'G', utt_id, text, gen_dir, pitch_gt, noise_outputs, None, None, phoneme, uv]))
                t.set_description(
                    f"Pred_shape: {outputs.shape}, gt_shape: {targets.shape}")
            else:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

        return {}


if __name__ == '__main__':
    FastSpeech2Task.start()
