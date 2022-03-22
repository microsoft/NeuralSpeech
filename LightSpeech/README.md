# LightSpeech
[LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search, ICASSP 2021](https://arxiv.org/abs/2102.04040), by Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Jinzhu Li, Sheng Zhao, Enhong Chen and Tie-Yan Liu, is a method to find lightweight and efficient TTS models with neural architecture search.

## Dependencies
```bash
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install libsndfile1 -y
```
Note: pytorch_lightning 0.6.0 may have a security issue(see [here](https://github.com/advisories/GHSA-r5qj-cvf9-p85h) and [here](https://github.com/PyTorchLightning/pytorch-lightning/pull/12212)), you can ignore it or try to solve it following this [patch](https://github.com/PyTorchLightning/pytorch-lightning/commit/8b7a12c52e52a06408e9231647839ddb4665e8ae).

## Quick Start

### 1. Prepare dataset

Note: You can also download our preprocessed binarized data from [here](https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/data.tgz), then you can skip step 1-3, and directly navigate to [step 4](#4-train-lightspeech).

```bash
mkdir -p data/raw/
cd data/raw/
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -zxf LJSpeech-1.1.tar.bz2
cd ../../
python datasets/tts/lj/prepare.py
```
### 2. Forced alignment
```bash
# Download MFA first: https://montreal-forced-aligner.readthedocs.io/en/stable/aligning.html
# unzip to montreal-forced-aligner
./montreal-forced-aligner/bin/mfa_train_and_align data/raw/LJSpeech-1.1/mfa_input data/raw/LJSpeech-1.1/dict_mfa.txt data/raw/LJSpeech-1.1/mfa_outputs -t ./montreal-forced-aligner/tmp -j 24
```

### 3. Build binary data
```bash
PYTHONPATH=. python datasets/tts/lj/gen.py --config configs/tts/lj/lightspeech.yaml
```

### 4. Train LightSpeech
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. python tasks/lightspeech.py --config configs/tts/lj/lightspeech.yaml --exp_name lightspeech
```

### 5. Download pre-trained vocoder
```bash
mkdir wavegan_pretrained
```
download `checkpoint-1000000steps.pkl`, `config.yml`, `stats.h5` from [here](https://drive.google.com/open?id=1XRn3s_wzPF2fdfGshLwuvNHrbgD0hqVS) to `wavegan_pretrained/`
   
### 6. Inference
 ```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/lightspeech.py --config configs/tts/lj/lightspeech.yaml --exp_name lightspeech --infer
```
The generated output will be under `checkpoints/lightspeech/generated_[step]` folder.

### 7. Trained Checkpoint
We release our trained model checkpoint [here](https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/model_ckpt_steps_84000.ckpt). You can donwnload it and place it under `checkpoints/lightspeech`. Then you can do the inference following [step 6](#6-inference).

You can refer to [https://speechresearch.github.io/lightspeech/](https://speechresearch.github.io/lightspeech/) for generated audio samples.

## Reference

If you find LightSpeech useful in your work, you can cite the paper as below:

    @inproceedings{luo2021lightspeech,
        title={Lightspeech: Lightweight and fast text to speech with neural architecture search},
        author={Luo, Renqian and Tan, Xu and Wang, Rui and Qin, Tao and Li, Jinzhu and Zhao, Sheng and Chen, Enhong and Liu, Tie-Yan},
        booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
        pages={5699--5703},
        year={2021},
        organization={IEEE}
    }

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct),
[trademark notice](https://docs.opensource.microsoft.com/releasing/), and [security reporting instructions](https://docs.opensource.microsoft.com/releasing/maintain/security/).
