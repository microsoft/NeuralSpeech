import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tts_utils.pl_utils import data_loader
import os, sys
from multiprocessing.pool import Pool
from tqdm import tqdm

from modules.tts_modules import DurationPredictorLoss
from tts_utils.hparams import hparams, set_hparams
from tts_utils.plot import plot_to_figure, numpy_to_figure, spec_numpy_to_figure
from tts_utils.world_utils import restore_pitch, process_f0

import numpy as np

from tasks.fs2 import FastSpeechDataset
from modules.priorgrad import PriorGrad
from tasks.transformer_tts import TransformerTtsTask
import time

import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import tts_utils

sys.path.append("hifi-gan")


class PriorGradTask(TransformerTtsTask):
    def __init__(self):
        super(PriorGradTask, self).__init__()
        self.dur_loss_fn = DurationPredictorLoss()
        self.mse_loss_fn = torch.nn.MSELoss()
        self.use_phone_stat = hparams['use_phone_stat'] if 'use_phone_stat' in hparams else False

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
        model = PriorGrad(arch, self.phone_encoder)
        print("encoder params:{}".format(sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)))
        print("decoder params:{}".format(sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)))
        return model

    def _training_step(self, sample, batch_idx, _):
        input = sample['src_tokens']  # [B, T_t]
        target = sample['targets']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        pitch = sample['pitch']
        energy = sample['energy']
        uv = sample['uv']

        # phoneme-level computed target statistics
        target_mean = sample['targets_mean'] if 'targets_mean' in sample else None
        target_std = sample['targets_std'] if 'targets_std' in sample else None

        # get mask for target beforehand for MAS
        # only get the first dim (previously repeated along mel bin)
        target_nonpadding = self.weights_nonzero_speech(target)[:, :, 0] # [B, T_mel]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        loss_output, output = self.run_model(self.model, input, mel2ph, spk_embed, target, target_mean, target_std, target_nonpadding,
                                             pitch=pitch, uv=uv, energy=energy, is_training=True,
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

        # phoneme-level computed target statistics
        target_mean = sample['targets_mean'] if 'targets_mean' in sample else None
        target_std = sample['targets_std'] if 'targets_std' in sample else None

        # get mask for target beforehand for MAS
        # only get the first dim (previously repeated along mel bin)
        target_nonpadding = self.weights_nonzero_speech(target)[:, :, 0] # [B, T_mel]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, input, mel2ph, spk_embed, target, target_mean, target_std, target_nonpadding,
                                                      pitch=pitch, uv=uv,
                                                      energy=energy, is_training=True,
                                                      return_output=True)

        outputs['total_loss'] = outputs['losses']['diffusion']
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']
        outputs = tts_utils.tensors_to_scalars(outputs)
        if batch_idx < 10:
            # run reverse diffusion for sampling
            # why only run reverse for subset?: DDPM takes long time for reverse (if we use 50 or so). too long to evaluate, not worth it
            # sample 10 points and monitor these spec losses as proxy
            outputs_reverse, model_out_reverse = self.run_model(self.model, input, mel2ph, spk_embed, target, target_mean, target_std, target_nonpadding,
                                                                pitch=pitch, uv=uv,
                                                                energy=energy, is_training=False,
                                                                return_output=True)
            if 'pitch_logits' in model_out_reverse:
                pitch[uv > 0] = -4
                pitch_pred = model_out_reverse['pitch_logits'][:, :, 0]
                pitch_pred[model_out_reverse['pitch_logits'][:, :, 1] > 0] = -4
                self.logger.experiment.add_figure(f'pitch_{batch_idx}', plot_to_figure({
                    'gt': pitch[0].detach().cpu().numpy(),
                    'pred': pitch_pred[0]
                        .detach().cpu().numpy()
                }), self.global_step)
            if 'mel_out' in model_out_reverse:
                mel_out = model_out_reverse['mel_out'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'mel_out_{batch_idx}', spec_numpy_to_figure(mel_out),
                                                  self.global_step)
            if 'mel' in outputs_reverse:
                outputs['losses']['mel'] = outputs_reverse['mel'].item()
            if 'encoder_proj_aligned' in model_out: # from MAS encoder_proj
                encoder_proj_aligned = model_out['encoder_proj_aligned'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'encoder_proj_aligned_{batch_idx}', spec_numpy_to_figure(encoder_proj_aligned),
                                                  self.global_step)
            # try plotting learned target mean & std
            if 'target_mean_aligned' in model_out_reverse:
                target_mean_aligned = model_out_reverse['target_mean_aligned'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'target_mean_{batch_idx}', spec_numpy_to_figure(target_mean_aligned),
                                                  self.global_step)
            if 'target_std_aligned' in model_out_reverse:
                target_std_aligned = model_out_reverse['target_std_aligned'][0].detach().cpu().numpy()
                self.logger.experiment.add_figure(f'target_std_{batch_idx}', spec_numpy_to_figure(target_std_aligned),
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

    def run_model(self, model, input, mel2ph, spk_embed, target, target_mean, target_std, target_nonpadding,
                  return_output=False, ref_mel='tgt', pitch=None, uv=None, energy=None, is_training=True):
        hparams['global_steps'] = self.global_step
        losses = {}
        if ref_mel == 'tgt':
            ref_mel = target

        output = model(input, mel2ph, spk_embed, ref_mel, target_mean, target_std, target_nonpadding, pitch, uv, energy, is_training)

        if is_training:
            # compute diffusion loss
            if self.use_phone_stat:
                losses['diffusion'] = self.scaled_mse_loss(output['noise_pred'], output['noise_target'], output['target_mean_aligned'], output['target_std_aligned'])
            else:
                losses['diffusion'] = self.mse_loss(output['noise_pred'], output['noise_target'])
        else:
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

    def scaled_mse_loss(self, decoder_output, target, target_mean, target_std):
        # inverse of diagonal matrix is 1/x for each element
        sigma_inv = torch.reciprocal(target_std)
        mse_loss = (((decoder_output - target) * sigma_inv) ** 2)
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

    # inference beta grid search implementation, not optimal but serves the purpose
    # grid search function modified from WaveGrad: https://github.com/ivanvovk/WaveGrad/blob/master/benchmark.py
    # BSD 3-Clause License
    # Copyright (c) 2020, Ivan Vovk, All rights reserved.
    def get_best_noise_schedule(self,src_tokens, mel2ph, spk_embed=None, ref_mels=None, target_mean=None, target_std=None,
                                target_nonpadding=None, pitch=None, uv=None, energy=None, is_training=True, fast_sampling=False,
                                skip_decoder=False, n_iter=6, betas_range_start=1e-4, betas_range_end=1e-1):
        assert ref_mels is not None, "we need target mel to search"
        # method to get the best inference noise schedule from the trained model.
        def generate_betas_grid(n_iter, betas_range):
            if n_iter > 12:
                return np.array([np.linspace(betas_range_start, betas_range_end, n_iter)])

            betas_range = np.log10(betas_range)
            exp_step = (betas_range[1] - betas_range[0]) / (n_iter - 1)
            exponents = 10 ** np.arange(betas_range[0], betas_range[1] + exp_step, step=exp_step)
            max_grid_size = None

            # hard-wired grid search spaces
            # max_grid_size is defined per n_iter to make good compromise between search speed and the quality of the noise schedule
            # too fine-grained max_grid_size does not improve the final audio quality much, but the search speed is way slower
            if n_iter == 2:
                exponents = np.array([1e-1, 1e-1])
                max_grid_size = 9 ** 2
            elif n_iter == 6:
                exponents = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e-1, 1e-1])
                max_grid_size = 9 ** 6
            elif n_iter == 12:
                exponents = np.array([1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-1])
                max_grid_size = 9 ** 9 # reasonable trade-off. one can increase to 9 ** 10 for more fine-grained search
            else:
                raise NotImplementedError("Not a valid --fast_iter. Only 2, 6, and 12 steps are supported for the grid search!")

            grid = []
            state = int(''.join(['1'] * n_iter))  # initial state
            final_state = 9 ** n_iter
            step = int(np.ceil(final_state / (max_grid_size)))

            print("generating {}-step inference schedules for grid search...")
            for _ in tqdm(range(max_grid_size)):
                multipliers = list(map(int, str(state)))
                # hard-wired rules
                if n_iter in [2, 3]:
                    if 0 in multipliers:
                        state += step
                        continue
                elif n_iter == 6:
                    if 0 in multipliers or multipliers[3] >= multipliers[4] or multipliers[4] >= multipliers[5]:
                        state += step
                        continue
                elif n_iter == 12:
                    if 0 in multipliers or multipliers[0] >= multipliers[1] or multipliers[2] >= multipliers[3] or\
                            multipliers[4] >= multipliers[5] or multipliers[5] >= multipliers[6] or multipliers[6] >= multipliers[7] or\
                            multipliers[8] >= multipliers[9] or multipliers[9] >= multipliers[10] or multipliers[10] >= multipliers[11]:
                        state += step
                        continue

                betas = [mult * exp for mult, exp in zip(multipliers, exponents)]
                grid.append(betas)
                state += step
            return grid

        grid = generate_betas_grid(n_iter, (betas_range_start, betas_range_end))
        grid_low = grid

        best_loss = 999
        best_grid = None

        for i in tqdm(range(len(grid_low))):
            # swap inference noise schedule
            self.model.decoder.params.inference_noise_schedule = grid_low[i]
            # get test loss
            with torch.no_grad():
                outputs = self.model(src_tokens, mel2ph, spk_embed, ref_mels,
                                     target_mean, target_std, target_nonpadding, pitch, uv, None,
                                     is_training=False, fast_sampling=hparams['fast'])
                mel_out = outputs['mel_out']
            loss = self.l1_loss(mel_out, ref_mels).item()
            # update best_grid based on best_loss
            if loss < best_loss:
                print("")
                print("better grid found! loss {} grid {}".format(loss, grid_low[i]))
                best_loss = loss
                best_grid = grid_low[i]

        print("best grid: {}".format(best_grid))
        best_schedule_name = 'betas'+str(hparams['fast_iter'])+'_'+hparams['work_dir'].split('/')[-1] + '_' + str(self.global_step)
        print("saving the best grid to {}".format(best_schedule_name))
        np.save(os.path.join(hparams['work_dir'], best_schedule_name), best_grid)

        self.model.decoder.params.inference_noise_schedule = best_grid

    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        input = sample['src_tokens']

        # phoneme-level computed target statistics. stats are based on the training set
        target_mean = sample['targets_mean'] if 'targets_mean' in sample else None
        target_std = sample['targets_std'] if 'targets_std' in sample else None

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

        # try to search for best noise schedule. if the best schedule exists, load and update the noise schedule
        if batch_idx == 0 and hparams['fast']:
            best_schedule_name = 'betas'+str(hparams['fast_iter'])+'_'+hparams['work_dir'].split('/')[-1] + '_' + str(self.global_step) + '.npy'
            if best_schedule_name not in os.listdir(hparams['work_dir']):
                print("INFO: searching for best {}-step inference beta schedule!".format(hparams['fast_iter']))
                mel2ph_for_search = sample['mel2ph']
                ref_mel_for_search = sample['targets']
                if hparams['dur'] == 'mas':
                    # get mask for target beforehand for MAS
                    # only get the first dim (previously repeated along mel bin)
                    target_nonpadding_for_search = self.weights_nonzero_speech(ref_mel_for_search)[:, :, 0]  # [B, T_mel]
                else:
                    target_nonpadding_for_search = None
                self.get_best_noise_schedule(input, mel2ph_for_search, spk_embed, ref_mel_for_search,
                                            target_mean, target_std, target_nonpadding_for_search, pitch, uv, None,
                                             is_training=False, fast_sampling=hparams['fast'],
                                            n_iter=hparams['fast_iter'], betas_range_start=hparams['diff_beta_start'], betas_range_end=hparams['diff_beta_end'])
            else:
                best_schedule = np.load(os.path.join(hparams['work_dir'], best_schedule_name))
                self.model.decoder.params.inference_noise_schedule = best_schedule
                print("INFO: saved noise schedule found in {}".format(os.path.join(hparams['work_dir'], best_schedule_name)))
                print("diffusion decoder inference noise schedule is reset to {}".format(best_schedule))
        if hparams['fast_iter'] > 12:
            print("WARNING: --fast_iter higher than 12 is provided. Grid search is disabled and will use the linearly spaced noise schedule!")
            print("WARNING: the quality is expected to be WORSE than the grid-searched noise schedule!")
            print("WARNING: the officially supported --fast_iter is 2, 6, or 12 steps!")

        with tts_utils.Timer('fs', print_time=hparams['profile_infer']):
            torch.cuda.synchronize()
            tic = time.time()
            outputs = self.model(input, mel2ph, spk_embed, None, target_mean, target_std, None, pitch, uv, None,
                                 is_training=False, fast_sampling=hparams['fast'])
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

            if hparams['fast']:
                gen_dir = os.path.join(hparams['work_dir'],
                                       f'generated_fast{hparams["fast_iter"]}_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            else:
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
    PriorGradTask.start()
