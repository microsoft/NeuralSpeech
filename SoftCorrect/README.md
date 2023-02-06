# SoftCorrect


[SoftCorrect: Error Correction with Soft Detection for Automatic Speech Recognition](https://arxiv.org/abs/2212.01039), by Yichong Leng, Xu Tan, Wenjie Liu, Kaitao Song, Rui Wang, Xiang-Yang Li, Tao Qin, Edward Lin and Tie-Yan Liu in *AAAI 2023*, is a novel non-autoregressive error correction method for automatic speech recognition with soft error detection. It uses an encoder trained with a language modeling loss to detect error tokens and a constrained decoder to correct errors.

## Dependencies
Currently we implement SoftCorrect on the [fairseq-0.10.1](https://github.com/pytorch/fairseq/tree/v0.10.1). Please refer to the fairseq installation.
Some dependencies are as follows:
- Python 3
- NumPy
- PyTorch==1.6.0
- fairseq==0.10.1

## SoftCorrect Modules
SoftCorrect consists an encoder as error detector and a decoder as error corrector, which can be trained seperately.

Since our model is a Chinese character-based model, arbitrary SentencePiece model can be used during the inference phase. We provide a SentencePiece model in [here](https://msramllasc.blob.core.windows.net/modelrelease/FastCorrect_zhwiki_sentencepiece.model), which is trained on Chinese wiki data.

### Error Detector (Encoder)

The error detector can be trained on pseudo data only, which only requires unpaired text data. We release the model trained on our internal unpaired text dataset (400M sentences). The unpaired text data can be easily obtained from wiki or other corpus. 

#### Generate the databin
After collecting the unpaired data, we binarize it with `runs/data_gen_unpaired.sh`

#### Train the BERT generator
We use a BERT generator to construct the pseudo data for ASR correction (sentence with errors simulated by BERT). We can train the BERT generator with `runs/train_bert_generator.sh`. The pretrained BERT model on the internal data can be downloaded in [here](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/bert_generator/checkpoint30.pt).

#### Train the error detector
Since SoftCorrect makes use of multiple candidates, we use BERT generator to construct the pseudo ASR correction data with multiple candidates. The data generation is achieved in an online manner. To stablize the training, the error detector can be trained firstly with `runs/pretrain_detector.sh` for 1 epoch (BERT style loss with multiple candidate as input) and then trained with `runs/finetune_detector.sh` (Anti-copy loss).
The model trained on two stage can be download in [first stage](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/detector_pretrain/checkpoint5.pt) and [second stage](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/detector_finetune/checkpoint16.pt). 

#### Test the error detector
The test data can be downloaded and unzipped from [here](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/eval_data.tar.gz). To test the error detector model after the second-stage training, we can score each token in each candidate with the error detector with `runs/test_detector.sh` and combine the score with the acoustic score with `runs/detect_with_lmscore.sh`. The results will be used by error corrector.

Note that the model after the second-stage training is the final error detector. We release the pretrained BERT generator and the first-stage model to help the potential future work based on SoftCorrect. And the first-stage training (for 1 epoch) is not a necessary step for SoftCorrect. It will be fine if the detector is trained directly with Anti-copy loss (second-stage).

### Error Corrector (Decoder)

#### Pretrained model

The error corrector is pretrained on unpaired text data with `runs/pretrain_corrector.sh`. The pretrained model can be downloaded from [here](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/corrector_pretrain/checkpoint6.pt).

#### Finetuning data generation

The pretrained model can be finetuned on AISHELL-1 data. The finetuning databin can be generate with `runs/data_gen_corrector_finetune.sh`. The raw data required by the script can be downloaded and unzipped in [here](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/raw_data.tar.gz). For the preparation of finetuning data, please refer to [Step 2 of FastCorrect](https://github.com/microsoft/NeuralSpeech/blob/master/FastCorrect/README.md) since the procedure is similar.

#### Finetuning model
We can finetune the pretrained corrector model with `runs/finetune_corrector.sh` (for about 10 epochs).

#### Test the error corrector
The test data can be downloaded and unzipped from [here](https://msramldl.blob.core.windows.net/modelrelease/softcorrect/eval_data.tar.gz) (If not downloaded in previous section). We test the error corrector with `runs/test_corrector.sh`.

After installing sctk (`./install_sctk.sh`), we can calculate WER with `cal_wer_aishell.sh`.

## Reference

If you find SoftCorrect useful in your work, you can cite the paper as below:

    @inproceedings{leng2023softcorrect,
        title={SoftCorrect: Error Correction with Soft Detection for Automatic Speech Recognition},
        author={Leng, Yichong and Tan, Xu and Liu, Wenjie and Song, Kaitao and Wang, Rui and Li, Xiang-Yang and Qin, Tao and Lin, Edward and Liu, Tie-Yan},
        booktitle={AAAI},
        year={2023}
    }

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct),
[trademark notice](https://docs.opensource.microsoft.com/releasing/), and [security reporting instructions](https://docs.opensource.microsoft.com/releasing/maintain/security/).

## Related Works


* [FastCorrect: Fast Error Correction with Edit Alignment for Automatic Speech Recognition](https://arxiv.org/abs/2105.03842)

* [FastCorrect 2: Fast Error Correction on Multiple Candidates for Automatic Speech Recognition](https://arxiv.org/abs/2109.14420.pdf)

