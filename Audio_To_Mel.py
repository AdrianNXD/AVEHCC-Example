
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class AudioToMelSpectrogram:
    def __init__(
        self,
        samplerate=48000,
        duration=2.0,
        output_size=(80, 80),
        n_mels=128,
        n_fft=2048,
        normalize=True,
        linear_scale=1.0,
    ):
        """
        Converts audio chunks to mel-spectrogram tensors.

        Parameters
        ----------
        samplerate : int
            sample rate of the audio material
        duration : float
            length of the audio chunk in seconds
        output_size : tuple[int, int]
            target resolution of the spectrogram
        n_mels : int
            number of mel-bins before resizing
        n_fft : int
            FFT window size
        normalize : bool
            True: dB-scale with fixed value range
            False: linear mels without chunk normalization
        linear_scale : float
            controls the fixed divisor used to scale linear mel spec values to the range [0, 1], independently of each audio chunk
            normalize=False. has to be > 0
        """

        if linear_scale <= 0:
            raise ValueError("linear_scale must be greater than 0.")

        self.samplerate = samplerate
        self.duration = duration
        self.output_size = output_size
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.normalize = normalize
        self.linear_scale = linear_scale

        clip_samples = int(samplerate * duration)
        n_frames = output_size[1]

        self.hop_length = max(
            1, (clip_samples - n_fft) // (n_frames - 1)
        )

        self.melspec_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=samplerate,
            n_fft=n_fft,
            hop_length=self.hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.db_converter = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80,
        )

    def __call__(self, audio):
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if not isinstance(audio, torch.Tensor):
            raise TypeError(
                "audio must be a numpy.ndarray or torch.Tensor."
            )

        audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        elif audio.ndim == 2:
            # (samples, channels) -> (channels, samples)
            audio = audio.transpose(0, 1)

        else:
            raise ValueError(
                f"Unexpected audio dimensions: {audio.shape}. "
                "Expected (samples,) or (samples, channels)."
            )

        # Convert multichannel audio to mono.
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Compute the Mel spectrogram.
        mel = self.melspec_converter(audio)

        if self.normalize:
            # Convert to dB and scale to [0, 1].
            mel = self.db_converter(mel)
            mel = (mel + 80.0) / 80.0
            mel = mel.clamp(0.0, 1.0)

        else:
            # Apply fixed linear scaling.
            mel = mel / self.linear_scale
            mel = mel.clamp(0.0, 1.0)

        # Resize to the target dimensions.
        mel = F.interpolate(
            mel.unsqueeze(1),
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )

        mel = mel.clamp(0.0, 1.0)

        # [1, 1, H, W] -> [1, H, W]
        mel = mel.squeeze(1)

        return mel
