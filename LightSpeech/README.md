# LightSpeech
[LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search, ICASSP 2021](https://arxiv.org/abs/2102.04040), by Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Jinzhu Li, Sheng Zhao, Enhong Chen and Tie-Yan Liu, is a method to find lightweight and efficient TTS models with neural architecture search.

## Dependencies
- Python=3.7
- packages:
```bash
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install libsndfile1 -y
```
Note: pytorch_lightning 0.6.0 may have a security issue(see [here](https://github.com/advisories/GHSA-r5qj-cvf9-p85h) and [here](https://github.com/PyTorchLightning/pytorch-lightning/pull/12212)), you can ignore it or try to solve it following this [patch](https://github.com/PyTorchLightning/pytorch-lightning/commit/8b7a12c52e52a06408e9231647839ddb4665e8ae).

## Quick Start

### 1. Prepare dataset

Note: You can download our preprocessed binarized data from [here](https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/data.tgz), and unpack it:
```bash
wget https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/data.tgz
tar -zxvf data.tgz
```
then you can skip step 1-3, and directly navigate to [step 4](#4-train-lightspeech).

```bash
mkdir -p data/raw/
cd data/raw/
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xf LJSpeech-1.1.tar.bz2
cd ../../
python datasets/tts/lj/prepare.py
```
### 2. Forced alignment
We use Montreal-Forced-Aligner v1.0.0. You can get it from [github link](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/tag/v1.0.0)
```bash
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
tar -zxvf montreal-forced-aligner_linux.tar.gz
./montreal-forced-aligner/bin/mfa_train_and_align data/raw/LJSpeech-1.1/mfa_input data/raw/LJSpeech-1.1/dict_mfa.txt data/raw/LJSpeech-1.1/mfa_outputs -t ./montreal-forced-aligner/tmp -j 24
```

### 3. Build binary data
```bash
PYTHONPATH=. python datasets/tts/lj/gen.py --config configs/tts/lj/lightspeech.yaml
```

### 4. Train LightSpeech
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # we run on 4 cards in our paper
export PYTHONPATH=.
python tasks/lightspeech.py \
    --config configs/tts/lj/lightspeech.yaml \
    --exp_name lightspeech \
    --reset
```
`--reset` is to reset the hyper parameters stored in the config file under the checkpoint folder if exists with the config file proved through `--config`.

### 5. Download pre-trained vocoder
```bash
mkdir wavegan_pretrained
```
download `checkpoint-1000000steps.pkl`, `config.yml`, `stats.h5` from [here](https://drive.google.com/open?id=1XRn3s_wzPF2fdfGshLwuvNHrbgD0hqVS) to `wavegan_pretrained/`
   
### 6. Inference
### Generate audio
To generate audio, run below command:
 ```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
python tasks/lightspeech.py \
    --config configs/tts/lj/lightspeech.yaml \
    --exp_name lightspeech \
    --reset \
    --infer
```
The generated output will be under `checkpoints/[exp_name]]/generated_[step]` folder.

### Measure inference time and RTF
To measure the inference time of the model, add `--hparams "profile_infer=True"` to the inference command. This will measure the inference time of the model (exclude the vocoder) along with the time of generated audio waves. You will see the model inference time `model_time` and the generated audio waves time `gen_wav_time` in the log output. After the inference is done, you can get the total model inference time and the total generated audio waves time. For example, following output log
```
model_time: 3.948242664337158
gen_wav_time: 626.7530158730159
```
means the total inference time of the model is 3.95 seconds, and the time of all the generateed audio waves is 626.75 seconds. To calculate the RTF, just divide the model inference time by the genearated audio waves time: `RTF = model_time/gen_wav_time`. In this example, the `RTF=3.95/626.75=0.0063`.

In our paper, we measure the inference time on CPU in single thread. To run in CPU mode, just set `CUDA_VISIBLE_DEVICES=`, and to use single thread, set `OMP_NUM_THREADS=1`. Then the command is:
```bash
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=.
python tasks/lightspeech.py \
    --config=configs/tts/lj/lightspeech.yaml \
    --exp_name=lightspeech \
    --infer \
    --reset \
    --hparams="profile_infer=True"
```
Then you will get the model inference time on CPU in single thread, and you can calculate the RTF. We report our result here:
```
model_time: 5.864823732376099
gen_wav_time: 626.7530158730159
```
So the RTF is `5.86/626.75=0.0093`.

### 7. Trained Checkpoint
We release our trained model checkpoint [here](https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/model_ckpt_steps_100000.ckpt). You can donwnload it and place it under `checkpoints/lightspeech`. Then you can do the inference following [step 6](#6-inference).
```bash
mkdir -p checkpoints/lightspeech
cd checkpoints/lightspeech
wget https://msramllasc.blob.core.windows.net/modelrelease/LightSpeech/model_ckpt_steps_100000.ckpt
cd ../../
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
python tasks/lightspeech.py \
    --config configs/tts/lj/lightspeech.yaml \
    --exp_name lightspeech \
    --reset \
    --infer
```

You can refer to [https://speechresearch.github.io/lightspeech/](https://speechresearch.github.io/lightspeech/) for generated audio samples.

### 8. Inference with User Input
`tasks/lightspeech_inference.py` provides the text-to-speech inference of LightSpeech with user input text file defined by `--inference_text`. Refer to `inference_text.txt` for an example.
```bash
# the following command performs text-to-speech inference from inference_text.txt
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
python tasks/lightspeech_inference.py \
    --config configs/tts/lj/lightspeech.yaml \
    --exp_name lightspeech \
    --reset \
    --inference_text inference_text.txt
```

Samples are saved to the folder `checkpoints/[exp_name]/inference_[inference_text]_[step]`.

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
