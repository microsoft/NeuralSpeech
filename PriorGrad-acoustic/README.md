## PriorGrad-acoustic

This repository is an official PyTorch implementation of the paper:

> Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, Tie-Yan Liu. "PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior." _ICLR_ (2022).
>[[arxiv]](https://arxiv.org/abs/2106.06406)
>

This repository contains an acoustic model (text-conditional mel-spectrogram synthesis) presented in PriorGrad.

## Demo

Refer to the [demo page](https://speechresearch.github.io/priorgrad/) for the samples from the model.

## Quick Start and Examples

1. Navigate to PriorGrad-acoustic root and initialize submodule ([HiFi-GAN](https://github.com/jik876/hifi-gan) vocoder)
   ```bash
   pip install -r requirements.txt
   git submodule init
   git submodule update
   ```

2. Prepare the dataset (LJspeech)
   ```bash
   mkdir -p data/raw/
   cd data/raw/
   wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
   tar -xvf LJSpeech-1.1.tar.bz2
   cd ../../
   python datasets/tts/lj/prepare.py
   ```
3. Forced alignment for duration predictor training
   ```bash
   # The following commands are tested on Ubuntu 18.04 LTS.
   sudo apt install libopenblas-dev libatlas3-base
   # Download MFA from https://montreal-forced-aligner.readthedocs.io/en/stable/aligning.html
   wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
   # Unzip to montreal-forced-aligner
   tar -zxvf montreal-forced-aligner_linux.tar.gz
   # See https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/149 regarding this fix
   cd montreal-forced-aligner/lib/thirdparty/bin && rm libopenblas.so.0 && ln -s ../../libopenblasp-r0-8dca6697.3.0.dev.so libopenblas.so.0
   cd ../../../../
   # Run MFA
   ./montreal-forced-aligner/bin/mfa_train_and_align data/raw/LJSpeech-1.1/mfa_input data/raw/LJSpeech-1.1/dict_mfa.txt data/raw/LJSpeech-1.1/mfa_outputs -t ./montreal-forced-aligner/tmp -j 24
   ```

4. Build binary data and extract mean & variance for PriorGrad-acoustic. The mel-spectrogram is compatible with open-source [HiFi-GAN](https://github.com/jik876/hifi-gan)

   ```bash
   PYTHONPATH=. python datasets/tts/lj/gen_fs2_p.py --config configs/tts/lj/priorgrad.yaml --exp_name priorgrad
   ```

5. Train PriorGrad-acoustic
   ```bash
   # the following command trains PriorGrad-acoustic with default parameters defined in configs/tts/lj/priorgrad.yaml
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py --config configs/tts/lj/priorgrad.yaml --exp_name priorgrad --reset
   ```

6. Download pre-trained HiFi-GAN vocoder
    ```
    mkdir hifigan_pretrained
    ```
    download `generator_v1`, `config.json` from [Google Drive](https://drive.google.com/drive/folders/1XtZ_AaYIsnx1zh_HxhrG5SZ6MUJV59gm) to `hifigan_pretrained/`

   
7. Inference (fast mode with T=12)
    ```bash
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py --config configs/tts/lj/priorgrad.yaml --exp_name priorgrad --reset \
    --infer --fast --fast_iter 12
    ```
   
when `--infer --fast`, the model applies grid search of beta schedules with the specified number of `--fast_iter` steps for the given model checkpoint.

2, 6, and 12 `--fast_iter` are officially supported. If the value higher than 12 is provided, the model uses a linear beta schedule. Note that the linear schedule is expected to perform worse.


`--infer` without `--fast` performs slow sampling with the same `T` as the forward diffusion used in training.
   

### Optional feature: Monotonic alignment search (MAS) support
Instead of MFA, FastSpeech 2 and PriorGrad also support Monotonic Alignment Search (MAS) used in [Glow-TTS](https://github.com/jaywalnut310/glow-tts/) for duration predictor training.

```cd monotonic_align && python setup.py build_ext --inplace && cd ..```

   ```bash
   # The following command trains a variant of PriorGrad which uses MAS for training the duration predictor.
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/priorgrad.py --config configs/tts/lj/priorgrad.yaml --hparams dur=mas --exp_name priorgrad_mas --reset
   ```

## Reference
If you find PriorGrad useful to your work, please consider citing the paper as below:

      @inproceedings{
      lee2022priorgrad,
      title={PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior},
      author={Lee, Sang-gil and Kim, Heeseung and Shin, Chaehun and Tan, Xu and Liu, Chang and Meng, Qi and Qin, Tao and Chen, Wei and Yoon, Sungroh and Liu, Tie-Yan},
      booktitle={International Conference on Learning Representations},
      year={2022},
      }

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct),
[trademark notice](https://docs.opensource.microsoft.com/releasing/), and [security reporting instructions](https://docs.opensource.microsoft.com/releasing/maintain/security/).