# BinauralGrad
[BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis](https://arxiv.org/abs/2205.14807), by Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo Mandic, Lei He, Xiang-Yang Li, Tao Qin, Sheng Zhao and Tie-Yan Liu, is a novel two-stage framework equipped with diffusion models to accurately predict the binaural audio waveforms. Our implement is based on the opensource code of [Diffwave](https://github.com/lmnt-com/diffwave). The audio samples can be found in the [demo page](https://speechresearch.github.io/binauralgrad/).

## Install

Prepare environment:
```
bash prepare_env.sh
```

## Training and Inference

### Data preparation
Before you start training, you'll need to prepare a training dataset. The training dataset used in paper can be download in [here](https://github.com/facebookresearch/BinauralSpeechSynthesis). We assume that the training dataset are in `data/trainset` (containing subject1, subject2, ...). Also, you need to prepare a test dataset. The test dataset used in paper can also be download in [here](https://github.com/facebookresearch/BinauralSpeechSynthesis). We assume that the test dataset are in `data/testset`  (containing subject1, subject2, ...).

You need to generate a geometric warpped binaural audio from the mono audio based on the position difference of left ear and right ear by running:
```
bash runs/geowarp_dataset.sh
```


### Training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.

Train the first stage model:
```
bash runs/train_stage_one.sh
```

Train the second stage model:
```
bash runs/train_stage_two.sh
```

The batch size in `src/binauralgrad/params.py` is 48, which works fine when trained on 8*V100(32G) GPUs and might be set smaller if trained on fewer or smaller GPUs.

The model can generate binaural audio with good quality trained with 300K steps, which could be further improved with 700K steps.

### Pretrained models

The pretrained model of two stages can be downloaded in [stage1](https://msramldl.blob.core.windows.net/modelrelease/binauralgrad/pretrained_ckpt.s1.pt) and [stage2](https://msramldl.blob.core.windows.net/modelrelease/binauralgrad/pretrained_ckpt.s2.pt). We assume the model of first stage is in `checkpoints/stage_one/pretrained_ckpt.s1.pt` and the model of second stage is in `checkpoints/stage_two/pretrained_ckpt.s2.pt`.

### Inference
After downloading and placing the pretrained models as mentioned in above section, we can test it by:

```
bash test.sh
```

The synthesized binaural audio will be in `checkpoints/stage_two/output_s2`. The object metric will be calculated and printed out in above script.

Note that we also require GPU with 32GB memory when inference. If you encounter out-of-memory error when using smaller GPU, you can smaller `clip_len` variable in `src/binauralgrad/inference.py`, Line 131.

## Reference

If you find BinauralGrad useful in your work, you can cite the paper as below:

    @inproceedings{leng2022binauralgrad,
        title={BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis},
        author={Leng, Yichong and Chen, Zehua and Guo, Junliang and Liu, Haohe and Chen, Jiawei and Tan, Xu and Mandic, Danilo and He, Lei and Li, Xiang-Yang and Qin, Tao and Zhao, Sheng and Liu, Tie-Yan},
        booktitle={Advances in Neural Information Processing Systems 36},
        year={2022}
    }

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct),
[trademark notice](https://docs.opensource.microsoft.com/releasing/), and [security reporting instructions](https://docs.opensource.microsoft.com/releasing/maintain/security/).
