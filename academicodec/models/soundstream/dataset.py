# 和 Encodec* 的 dataset.py 有点类似但是不完全一样
# 主要是 prob > 0.7 的时候多了 ans2
import glob
import random

import torch
import soundfile as sf
from torch.utils.data import Dataset


def _load_audio(filepath):
    """Load audio using soundfile, returns (waveform, sample_rate).
    waveform shape: (channels, samples) as float32 tensor."""
    data, sr = sf.read(filepath, dtype="float32")
    # soundfile returns (samples,) for mono or (samples, channels) for multi-channel
    waveform = torch.from_numpy(data)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    else:
        waveform = waveform.T  # (channels, samples)
    return waveform, sr


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir, use_spec_augment=False):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir + "/**/*.wav", recursive=True))
        print(len(self.filenames))
        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No .wav files found in {audio_dir} (searched recursively)")
        _, self.sr = _load_audio(self.filenames[0])
        self.max_len = 24000  # 24000
        self.use_spec_augment = use_spec_augment

    def _apply_spec_augment(self, waveform):
        # waveform: (channels, length)
        # We use a context manager to avoid storing gradients, just in case
        with torch.no_grad():
            n_fft = 1024
            hop_length = 256
            window = torch.hann_window(n_fft).to(waveform.device)
            
            stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                              window=window, return_complex=True)
            
            # SpecAugment parameters
            time_mask_param = max(1, stft.shape[-1] // 5)
            freq_mask_param = max(1, stft.shape[-2] // 5)
            
            # Time mask
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, max(1, stft.shape[-1] - t))
            stft[..., :, t0 : t0 + t] = 0.0
            
            # Freq mask
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(1, stft.shape[-2] - f))
            stft[..., f0 : f0 + f, :] = 0.0
            
            aug_waveform = torch.istft(stft, n_fft=n_fft, hop_length=hop_length, 
                                       window=window, length=waveform.shape[-1])
        return aug_waveform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print(self.filenames[index])
        prob = random.random()  # (0,1)
        if prob > 0.7:
            # data augmentation
            ans1 = torch.zeros(1, self.max_len)
            ans2 = torch.zeros(1, self.max_len)
            audio1 = _load_audio(self.filenames[index])[0]
            index2 = random.randint(0, len(self.filenames) - 1)
            audio2 = _load_audio(self.filenames[index2])[0]
            if audio1.shape[1] > self.max_len:
                st = random.randint(0, audio1.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans1 = audio1[:, st:ed]
            else:
                ans1[:, :audio1.shape[1]] = audio1
            if audio2.shape[1] > self.max_len:
                st = random.randint(0, audio2.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans2 = audio2[:, st:ed]
            else:
                ans2[:, :audio2.shape[1]] = audio2
            ans = ans1 + ans2
            return ans
        else:
            ans = torch.zeros(1, self.max_len)
            audio = _load_audio(self.filenames[index])[0]
            if audio.shape[1] > self.max_len:
                st = random.randint(0, audio.shape[1] - self.max_len - 1)
                ed = st + self.max_len
                ans = audio[:, st:ed]
            else:
                ans[:, :audio.shape[1]] = audio
        
        if getattr(self, 'use_spec_augment', False):
            # Apply SpecAugment with 50% probability
            if random.random() > 0.5:
                ans = self._apply_spec_augment(ans)

        return ans
