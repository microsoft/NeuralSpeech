# VideoDubber
[VideoDubber: Machine Translation with Speech-Aware Length Control for Video Dubbing](https://arxiv.org/abs/2211.16934), by Yihan Wu, Junliang Guo, Xu Tan, Chen Zhang, Bohan Li, Ruihua Song, Lei He, Sheng Zhao, Arul Menezes, Jiang Bian, is a novel machine translation framework with speech length control for video dubbing. Our implement is based on the opensource code of [Fairseq](https://github.com/facebookresearch/fairseq). The video samples can be found in the [demo page](https://speechresearch.github.io/videodubbing/).

## Install

Prepare envirommemt:
```
# Install Moses
git clone https://github.com/moses-smt/mosesdecoder.git

# Install Subword Neural Machine Translation which contains preprocessing scripts to segment text into subword units.
git clone https://github.com/rsennrich/subword-nmt.git

# Install jieba for Chinese text segmentation
pip install jieba

# Install fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

```

## Quick Start

prepare your own data and train your model following this command: 

```
bash scripts/train_cvss_c_zhen.sh
```

## Inference

After downloading the pretrained models , we can test it by:

```
bash scripts/infer_cvssc_zhen.sh
```

## Reference

If you find VideoDubber useful in your work, you can cite the paper as below:

    @misc{wu2022videodubber,
      title={VideoDubber: Machine Translation with Speech-Aware Length Control for Video Dubbing}, 
      author={Yihan Wu and Junliang Guo and Xu Tan and Chen Zhang and Bohan Li and Ruihua Song and Lei He and Sheng Zhao and Arul Menezes and Jiang Bian},
      year={2022},
      eprint={2211.16934},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
    }

    
