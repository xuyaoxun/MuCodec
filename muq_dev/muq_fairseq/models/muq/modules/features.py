import torchaudio
from torch import nn
import torch


class MelSTFT(nn.Module):
    def __init__(
        self,
        sample_rate=24000,
        n_fft=2048,
        hop_length=240,
        n_mels=128,
        is_db=False,
    ):
        super(MelSTFT, self).__init__()

        # spectrogram
        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # amplitude to decibel
        self.is_db = is_db
        if is_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, waveform):
        if self.is_db:
            return self.amplitude_to_db(self.mel_stft(waveform))
        else:
            return self.mel_stft(waveform)


class CQTPreprocessor(nn.Module):
    def __init__(self, 
            sr=24000,
            hop=960,
            nb=84,
            to_db = True,
        ) -> None:
        super().__init__()

        from nnAudio.features.cqt import CQT
        import torchaudio
        self.cqt_fn = CQT(
                            sr=sr, 
                            hop_length=hop,
                            n_bins=nb, 
                            fmin=32.7 if nb == 84 else 27.5, # 84 or 88
                            bins_per_octave=12, 
                            filter_scale=1, 
                            norm=1, 
                            window='hann', 
                            center=True, 
                            pad_mode='constant', 
                            trainable=False, 
                            output_format='Magnitude', 
                            verbose=True,
                        )
        if to_db:
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        else:
            self.amplitude_to_db = lambda x:x

    @torch.no_grad()
    def __call__(self, x):
        return self.amplitude_to_db(self.cqt_fn(x))