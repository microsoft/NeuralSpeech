# PromptTTS 2
[PromptTTS 2: Describing and Generating Voices with Text Prompt](https://arxiv.org/abs/2309.02285), by Yichong Leng, Zhifang Guo, Kai Shen, Xu Tan, Zeqian Ju, Yanqing Liu, Yufei Liu, Dongchao Yang, Leying Zhang, Kaitao Song, Lei He, Xiang-Yang Li, Sheng Zhao, Tao Qin, Jiang Bian, is a novel framework for voice generation conditioned on text prompt, facial image or other information. The audio samples can be found in the [demo page](https://speechresearch.github.io/prompttts2).

In this repo, we release code of the variation network (See `variation_network` folder) and the prompt generation pipeline (See `prompt_generation_pipeline` folder). 

## Reference

If you find PromptTTS 2 useful in your work, you can cite the papers as below:

    @inproceedings{
        leng2024prompttts2,
        title={PromptTTS 2: Describing and Generating Voices with Text Prompt},
        author={Yichong Leng and Zhifang Guo and Kai Shen and Xu Tan and Zeqian Ju and Yanqing Liu and Yufei Liu and Dongchao Yang and Leying Zhang and Kaitao Song and Lei He and Xiang-Yang Li and Sheng Zhao and Tao Qin and Jiang Bian},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
    }

    @inproceedings{
        shen2023naturalspeech,
        title={NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers}, 
        author={Kai Shen and Zeqian Ju and Xu Tan and Yanqing Liu and Yichong Leng and Lei He and Tao Qin and Sheng Zhao and Jiang Bian},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
    }


## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct),
[trademark notice](https://docs.opensource.microsoft.com/releasing/), and [security reporting instructions](https://docs.opensource.microsoft.com/releasing/maintain/security/).
