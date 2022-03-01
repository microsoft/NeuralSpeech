from multiprocessing.pool import Pool

import matplotlib

from tts_utils.pl_utils import data_loader

matplotlib.use('Agg')
from tts_utils.tts_utils import GeneralDenoiser
import glob
import os, sys
import re
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from modules import transformer_tts
from tasks.base_task import BaseTask, BaseDataset
from tts_utils.hparams import hparams
from tts_utils.indexed_datasets import IndexedDataset
from tts_utils.text_encoder import TokenTextEncoder
import json

import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import tts_utils
import logging
from tts_utils import audio

sys.path.append("hifi-gan")

class TransTTSDataset(BaseDataset):
    def __init__(self, data_dir, phone_encoder, prefix, hparams, shuffle=False):
        super().__init__(data_dir, prefix, hparams, shuffle)
        self.phone_encoder = phone_encoder
        self.data = None
        self.idx2key = np.load(f'{self.data_dir}/{self.prefix}_all_keys.npy')
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.train_idx2key = np.load(f'{self.data_dir}/train_all_keys.npy')
        self.use_indexed_ds = hparams['indexed_ds']
        self.indexed_bs = None

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
        key = self.idx2key[index]
        item = self._get_item(index)
        spec = torch.Tensor(item['mel'])[:self.hparams['max_frames']]
        sample = {
            "id": index,
            "utt_id": key,
            "text": item['txt'],
            "source": torch.LongTensor(item['phone']),
            "target": spec,
        }
        # if self.num_spk > 1:
        #     sample["spk_id"] = item['spk_id']
        #     sample["spk_embed"] = item['spk_embed']
        return sample

    def collater(self, samples):
        pad_idx = self.phone_encoder.pad()
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        utt_id = [s['utt_id'] for s in samples]
        text = [s['text'] for s in samples]
        src_tokens = tts_utils.collate_1d([s['source'] for s in samples], pad_idx)
        target = tts_utils.collate_2d([s['target'] for s in samples], pad_idx)
        prev_output_mels = tts_utils.collate_2d([s['target'] for s in samples], pad_idx, shift_right=True)
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        target_lengths = torch.LongTensor([s['target'].shape[0] for s in samples])
        target_lengths, sort_order = target_lengths.sort(descending=True)
        target = target.index_select(0, sort_order)
        prev_output_mels = prev_output_mels.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)
        src_lengths = src_lengths.index_select(0, sort_order)
        id = id.index_select(0, sort_order)
        utt_id = [utt_id[i] for i in sort_order]
        text = [text[i] for i in sort_order]
        ntokens = sum(len(s['source']) for s in samples)
        nmels = sum(len(s['target']) for s in samples)

        batch = {
            'id': id,
            'utt_id': utt_id,
            'nsamples': len(samples),
            'ntokens': ntokens,
            'nmels': nmels,
            'text': text,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'targets': target,
            'target_lengths': target_lengths,
            'prev_output_mels': prev_output_mels
        }
        return batch

    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS', 1))


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


class TransformerTtsTask(BaseTask):
    def __init__(self, *args, **kwargs):
        self.arch = hparams['arch']
        if isinstance(self.arch, str):
            self.arch = list(map(int, self.arch.strip().split()))
        if self.arch is not None:
            self.num_heads = tts_utils.get_num_heads(self.arch[hparams['enc_layers']:])
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        super().__init__(*args, **kwargs)

    @data_loader
    def train_dataloader(self):
        train_dataset = TransTTSDataset(hparams['data_dir'], self.phone_encoder, 'train', hparams, shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = TransTTSDataset(hparams['data_dir'], self.phone_encoder, 'valid', hparams, shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        if not hparams['use_training']:
            test_dataset = TransTTSDataset(hparams['data_dir'], self.phone_encoder, 'test', hparams, shuffle=False)
            self.test_dl = self.build_dataloader(test_dataset, False, self.max_eval_tokens,
                                                 self.max_eval_sentences)
        else:
            train_dataset = TransTTSDataset(hparams['data_dir'], self.phone_encoder, 'train', hparams, shuffle=True)
            self.test_dl = self.build_dataloader(train_dataset, True, self.max_eval_tokens,
                                                 self.max_eval_sentences)
        return self.test_dl

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False):
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = torch.cuda.device_count()

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= torch.cuda.device_count()
        if max_sentences is not None:
            max_sentences *= torch.cuda.device_count()
        indices = dataset.ordered_indices()
        batch_sampler = tts_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list)

    def build_model(self):
        arch = self.arch
        model = transformer_tts.TransformerTTS(arch, self.phone_encoder)
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
        input = sample['src_tokens']
        prev_output_mels = sample['prev_output_mels']
        target = sample['targets']
        output, _ = self.model(input, prev_output_mels, target)
        loss_output = self.loss(output, target)
        total_loss = sum(loss_output.values())
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
        prev_output_mels = sample['prev_output_mels']
        target = sample['targets']
        output, attn_logits = self.model(input, prev_output_mels, target)
        outputs = {}
        outputs['losses'] = self.loss(output, target)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nmels'] = sample['nmels']
        outputs['nsamples'] = sample['nsamples']

        src_lengths = sample['src_lengths']  # - 1 # exclude eos
        target_lengths = sample['target_lengths']
        src_padding_mask = input.eq(0)  # | input.eq(self.eos_idx)  # also exclude eos
        src_seg_mask = input.eq(self.seg_idx)
        target_padding_mask = target.abs().sum(-1).eq(0)

        encdec_attn = tts_utils.select_attn(attn_logits)
        outputs['focus_rate'] = tts_utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask).mean()
        outputs['phone_coverage_rate'] = tts_utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask,
                                                                           target_padding_mask).mean()
        attn_ks = src_lengths.float() / target_lengths.float()
        outputs['diagonal_focus_rate'], diag_mask = tts_utils.get_diagonal_focus_rate(encdec_attn, attn_ks,
                                                                                      target_lengths,
                                                                                      src_padding_mask,
                                                                                      target_padding_mask)
        outputs['diagonal_focus_rate'] = outputs['diagonal_focus_rate'].mean()
        outputs = tts_utils.tensors_to_scalars(outputs)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': tts_utils.AvgrageMeter(),
            'fr': tts_utils.AvgrageMeter(),
            'pcr': tts_utils.AvgrageMeter(),
            'dfr': tts_utils.AvgrageMeter(),
        }

        for output in outputs:
            n = output['nmels']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = tts_utils.AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
            all_losses_meter['fr'].update(output['focus_rate'], output['nsamples'])
            all_losses_meter['pcr'].update(output['phone_coverage_rate'], output['nsamples'])
            all_losses_meter['dfr'].update(output['diagonal_focus_rate'], output['nsamples'])
        return {f'{k}': round(v.avg, 4) for k, v in all_losses_meter.items()}

    def test_step(self, sample, batch_idx):
        logging.info('inferring batch {} with {} samples'.format(batch_idx, sample['nsamples']))
        with tts_utils.Timer('trans_tts', print_time=hparams['profile_infer']):
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
        # encdec_attn_logits = input.new(bsz, hparams['dec_layers'], 0, max_input_len).fill_(0).float()
        attn_pos = input.new(bsz).fill_(0).int()
        use_masks = []
        for i in range(hparams['dec_layers']):
            use_masks.append(input.new(self.num_heads[i]).fill_(0).float())
        # use_masks = input.new(hparams['dec_layers']*2).fill_(0).float()

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

            decoder_output, attn_logits = model.forward_decoder(decoder_input[:, :step + 1], encoder_out,
                                                                encoder_padding_mask,
                                                                incremental_state=incremental_state)
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
                                                                                               keepdim=True).int() - 5) & (
                                        torch.sigmoid(stop_logit) > 0.5)
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

                prev_weights_mask1 = tts_utils.sequence_mask(
                    torch.max(attn_pos - 1, attn_pos.new(attn_pos.size()).fill_(0)).float(),
                    encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                prev_weights_mask2 = 1.0 - tts_utils.sequence_mask(attn_pos.float() + 4,
                                                                   encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                enc_dec_attn_constraint_masks = []
                for i in range(hparams['dec_layers']):
                    mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[i][None, :,
                                                                                   None]  # bsz x head x L_kv
                    enc_dec_attn_constraint_masks.append(mask)
                # enc_dec_attn_constraint_masks = (prev_weights_mask1 + prev_weights_mask2)[:, None, None, :] * use_masks[None, :, None, None] # bsz x (n_layers x head) x 1 x L_kv

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
                    # prev_weights = (prev_weights * use_masks[None, :, None, None]).sum(dim=1) / use_masks.sum()
                    move_on = (prev_weights[:, -3:].mean(dim=1).gather(1, attn_pos[:, None].long())).squeeze(-1) < 0.7
                    move_on &= torch.argmax(prev_weights[:, -1], -1) > attn_pos.long()
                    return attn_pos + move_on.int()

                if step > 3 and stage == 1:
                    attn_pos = should_move_on()

        # size = encdec_attn_logits.size()
        # encdec_attn_logits = encdec_attn_logits.view(size[0], size[1]*size[2], size[3], size[4])
        encdec_attn = tts_utils.select_attn(encdec_attn_logits)

        if not hparams['profile_infer']:
            src_lengths = sample['src_lengths'] - 1  # exclude eos
            target_lengths = (1.0 - hit_eos[:, 1:].float()).sum(dim=-1) + 1
            src_padding_mask = input.eq(0) | input.eq(self.eos_idx)  # also exclude eos
            src_seg_mask = input.eq(self.seg_idx)
            target_padding_mask = decoded_mel.abs().sum(-1).eq(0)
            focus_rate = tts_utils.get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
            phone_coverage_rate = tts_utils.get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask,
                                                                    target_padding_mask)
            attn_ks = src_lengths.float() / target_lengths.float()
            diagonal_focus_rate, diag_mask = tts_utils.get_diagonal_focus_rate(encdec_attn, attn_ks, target_lengths,
                                                                               src_padding_mask,
                                                                               target_padding_mask)
        else:
            focus_rate, phone_coverage_rate, diagonal_focus_rate = None, None, None
        return decoded_mel, encdec_attn.unsqueeze(
            1), hit_eos, stop_logits, focus_rate, phone_coverage_rate, diagonal_focus_rate

    def estimate_decode_length(self, input_length):
        return input_length * 5 + 100

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
        if self.saving_result_pool is None:
            self.saving_result_pool = Pool(8)
            self.saving_results_futures = []
        self.prepare_vocoder()
        predictions = tts_utils.unpack_dict_to_list(predictions)
        for num_predictions, prediction in enumerate(tqdm(predictions)):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            utt_id = prediction.get('utt_id')
            text = prediction.get('text')
            src_tokens = prediction.get('src_tokens')
            src_lengths = prediction.get('src_lengths')
            targets = prediction.get("targets")
            outputs = prediction["outputs"]
            str_phs = self.phone_encoder.decode(src_tokens, strip_eos=True, strip_padding=True)
            targets = self.remove_padding(targets)  # speech
            outputs = self.remove_padding(outputs)
            pitch = None

            if 'encdec_attn' in prediction:
                encdec_attn = prediction['encdec_attn']
                encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
                encdec_attn = encdec_attn.T[:src_lengths, :len(targets)]
            else:
                encdec_attn = None

            gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}')
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/spec_plot', exist_ok=True)
            os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)

            wav_pred = self.inv_spec(outputs)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, outputs, f'P', utt_id, text, gen_dir, None, None, encdec_attn, str_phs]))

            wav_gt = self.inv_spec(targets)
            if targets is not None:
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, targets, 'G', utt_id, text, gen_dir]))

            if hparams['profile_infer']:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

    @staticmethod
    def save_result(wav_out, mel, prefix, utt_id, text, gen_dir,
                    pitch=None, noise_spec=None, alignment=None, str_phs=None,
                    phoneme=None, uv=None):
        base_fn = f'[{prefix}][{utt_id}]'
        base_fn += text.replace(":", "%3A")[:80]
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        audio.plot_spec(mel.T, f'{gen_dir}/spec_plot/{base_fn}.png')
        with open(f'{gen_dir}/text/{base_fn}.txt', 'w') as f:
            f.write(text)
        torch.save(mel.T, f'{gen_dir}/spec/{base_fn}.pt')
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
        if phoneme is not None:
            torch.save(phoneme, f'{gen_dir}/phoneme/{base_fn}.pt')
        if uv is not None:
            torch.save(uv, f'{gen_dir}/uv/{base_fn}.pt')

    def test_end(self, outputs):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    ##########
    # utils
    ##########
    def remove_padding(self, x, padding_idx=0):
        return tts_utils.remove_padding(x, padding_idx)

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        # target : B x T x mel
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - tts_utils.sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r

    def weighted_cross_entropy_with_logits(self, targets, logits, pos_weight=1):
        x = logits
        z = targets
        q = pos_weight
        l = 1 + (q - 1) * z
        return (1 - z) * x + l * (torch.log(1 + torch.exp(-x.abs())) + F.relu(-x))


if __name__ == '__main__':
    TransformerTtsTask.start()
