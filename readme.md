# MuCodec: Ultra Low-Bitrate Music Codec

This repository is the official code repository for MuCodec: Ultra Low-Bitrate Music Codec. You can find our paper on [arXiv] (https://arxiv.org/pdf/2409.13216). The demo page is available [here](https://xuyaoxun.github.io/MuCodec_demo/).

## Installation

You can install the necessary dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Inference

To run inference, use the following command:

```bash
python3 generate.py
```

We have provided a sample song `test.wav`, randomly sampled from the Million Song Dataset, in the `test_wav` folder. The default input path is `test_wav/test.wav`, and the output path for the reconstructed audio is `reconstruct/test.wav`.

In the `generate.py` file, we have implemented several functions to facilitate the music compression and reconstruction process. You can easily obtain compressed tokens from audio using the `sound2code` function, and reconstruct the audio from tokens using the `code2sound` function.

## Note

Please note that the open-sourced model was trained solely on the Million Song Dataset. Consequently, its performance on Chinese songs might not be as good as the model described in our paper. To achieve better results with Chinese songs, you can train the model with your own Chinese music dataset.

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