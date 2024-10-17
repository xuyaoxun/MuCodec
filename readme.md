# MuCodec: Ultra Low-Bitrate Music Codec

This repository is the official code repository for MuCodec: Ultra Low-Bitrate Music Codec. You can find our paper on [arXiv] (https://arxiv.org/pdf/2409.13216). The demo page is available [here](https://xuyaoxun.github.io/MuCodec_demo/).

In this repository, we provide the Mucodec model, inference scripts, and the checkpoint that has been trained on the Million Song Dataset. Specifically, we have released the model and inference code corresponding to the lowest bitrate of 0.35 kbps as mentioned in the paper, to demonstrate the effectiveness of our work. If you wish to experiment with music reconstruction at different bitrates, you can easily modify the parameters of the Residual Vector Quantizer (RVQ) and train the corresponding model.


MuCodec supports 48kHz, dual-channel (stereo) audio reconstruction. If the original audio is in a different format, it will first be converted to 48kHz, dual-channel audio.

## Installation

You can install the necessary dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Due to storage limitations, we have saved the model checkpoints on Hugging Face at https://huggingface.co/yaoxunxu/mucodec. You can easily download the models from Hugging Face and save them in the following directories:

- Save `audioldm_48k.pth` in the `tools` folder.
- Save `muq.pt` in the `muq_dev` folder.
- Save `mucodec.pt` in the `ckpt` folder.

Please note that all three checkpoints must be downloaded completely for the model to load correctly. The final file paths should be:

```
tools/audioldm_48k.pth
muq_dev/muq.pt
ckpt/mucodec.pt
```

## Inference

To run inference, use the following command:

```bash
python3 generate.py
```

We have provided a sample song `test.wav`, randomly sampled from the Million Song Dataset, in the `test_wav` folder. The default input path is `test_wav/test.wav`, and the output path for the reconstructed audio is `reconstruct/test.wav`.

In the `generate.py` file, we have implemented several functions to facilitate the music compression and reconstruction process. You can easily obtain compressed tokens from audio using the `sound2code` function, and reconstruct the audio from tokens using the `code2sound` function.

## Note

Please note that the open-sourced model was trained solely on the Million Song Dataset. Considering the quality issues of this dataset, the open-sourced model may not achieve the same performance as demonstrated in the demo. Unfortunately, due to copyright restrictions, we are unable to release the checkpoints trained on additional datasets. However, you can use your own dataset to further train the model and achieve better results.

## License

The code in this repository is released under the MIT license as found in the [LICENSE](LICENSE) file.

The model weights in this repository are released under the CC-BY-NC 4.0 license as found in the [LICENSE_weights](LICENSE_weights) file.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{xu2024mucodec,
  title={MuCodec: Ultra Low-Bitrate Music Codec},
  author={Xu, Yaoxun and Chen, Hangting and Yu, Jianwei and Tan, Wei and Gu, Rongzhi and Lei, Shun and Lin, Zhiwei and Wu, Zhiyong},
  journal={arXiv preprint arXiv:2409.13216},
  year={2024}
}
```