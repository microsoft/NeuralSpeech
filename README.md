# NeuralSpeech

**NeuralSpeech** is a research project at Microsoft Research Asia, which focuses on neural network based speech processing, including automatic speech recognition (ASR), text-to-speech synthesis (TTS), spatial audio synthesis, video dubbing, etc. 

Currently this repo covers several research work: 
* Automatic Speech Recognition
  + [FastCorrect, NeurIPS 2021](https://arxiv.org/abs/2105.03842) 
  + [FastCorrect 2, EMNLP 2021 Findings](https://arxiv.org/abs/2109.14420)
  + [SoftCorrect, AAAI 2023]() 
  + [MaskCorrect, EMNLP 2022](https://arxiv.org/abs/2211.13252)
  + [CMatch for ASR, INTERSPEECH 2021](https://arxiv.org/abs/2104.07491)
  + [Adapter for ASR, TASLP 2022](https://arxiv.org/abs/2105.11905)
* Text-to-Speech Synthesis
  + [LightSpeech, ICASSP 2021](https://arxiv.org/abs/2102.04040)
  + [PriorGrad, ICLR 2022](https://arxiv.org/abs/2106.06406)
* Spatial Audio Synthesis
  + [BinauralGrad, NeurIPS 2022](https://arxiv.org/abs/2205.14807)
* Video Dubbing
  + [VideoDubber, AAAI 2023](https://arxiv.org/abs/2211.16934)


For more research in NeuralSpeech project, you can refer to this page: https://speechresearch.github.io/. We will release more research work in the future. 

For our research on AI music, you can refer to our Muzic project: https://github.com/microsoft/muzic.


### We are hiring! 
We are hiring researchers on **speech (speech synthesis, speech recognition, voice conversion, audio processing), natural language processing, and machine learning**. Please contact Xu Tan (xuta@microsoft.com) if you have interests. 

## Reference

If you find NeuralSpeech project useful in your work, you can cite the following papers:

* [1] ***FastCorrect**: Fast Error Correction with Edit Alignment for Automatic Speech Recognition*, Yichong Leng, Xu Tan, Linchen Zhu, Jin Xu, Renqian Luo, Linquan Liu, Tao Qin, Xiang-Yang Li, Ed Lin and Tie-Yan Liu, **NeurIPS 2021**.
* [2] ***FastCorrect 2**: Fast Error Correction on Multiple Candidates for Automatic Speech Recognition*, Yichong Leng, Xu Tan, Rui Wang, Linchen Zhu, Jin Xu, Wenjie Liu, Linquan Liu, Tao Qin, Xiang-Yang Li, Ed Lin, Tie-Yan Liu, **Findings of EMNLP 2021**.
* [3] ***SoftCorrect**: Error Correction with Soft Detection for Automatic Speech Recognition*, Yichong Leng, Xu Tan, Wenjie Liu, Kaitao Song, Rui Wang, Xiang-Yang Li, Tao Qin, Edward Lin, Tie-Yan Liu, **AAAI 2023**.
* [4] ***[MaskCorrect]** Mask the Correct Tokens: An Embarrassingly Simple Approach for Error Correction*, Kai Shen, Yichong Leng, Xu Tan, Siliang Tang, Yuan Zhang, Wenjie Liu, Edward Lin, **EMNLP 2022**.
* [5] ***[CMatch]*** *Cross-domain Speech Recognition with Unsupervised Character-level Distribution Matching*, Wenxin Hou, Jindong Wang, Xu Tan, Tao Qin, Takahiro Shinozaki, **INTERSPEECH 2021**.
* [6] ***[Adapter]*** *Exploiting Adapters for Cross-lingual Low-resource Speech Recognition*, Wenxin Hou, Han Zhu, Yidong Wang, Jindong Wang, Tao Qin, Renjun Xu, Takahiro Shinozaki. **IEEE/ACM TASLP 2022**.
* [7] ***LightSpeech**: Lightweight and Fast Text to Speech with Neural Architecture Search*, Renqian Luo, Xu Tan, Rui Wang, Tao Qin, Jinzhu Li, Sheng Zhao, Enhong Chen and Tie-Yan Liu, **ICASSP 2021**.
* [8] ***PriorGrad**: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior*, Sang-gil Lee, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, Tie-Yan Liu, **ICLR 2022**.
* [9] ***BinauralGrad**: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis*, Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo Mandic, Lei He, Xiang-Yang Li, Tao Qin, Sheng Zhao and Tie-Yan Liu, **NeurIPS 2022**.
* [10] ***VideoDubber**: Machine Translation with Speech-Aware Length Control for Video Dubbing, Yihan Wu*, Junliang Guo, Xu Tan, Chen Zhang, Bohan Li, Ruihua Song,
Lei He, Sheng Zhao, Arul Menezes, Jiang Bian, **AAAI 2022**.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
