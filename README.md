# SoundStream for Pytorch

Unofficial [SoundStream](https://arxiv.org/abs/2107.03312) implementation of Pytorch with training code and 16kHz pretrained checkpoint.

[16kHz pretrained model](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/soundstream_16khz-20230425.ckpt) was trained on LibriSpeech train-clean-100 with NVIDIA T4 for about 150 epochs (around 50 hours) in total. The model is not causal.

```python
import torchaudio
import torch

model = torch.hub.load("kaiidams/soundstream-pytorch", "soundstream_16khz")
x, sr = torchaudio.load('input.wav')
x, sr = torchaudio.functional.resample(x, sr, 16000), 16000
with torch.no_grad():
    y = model.encode(x)
    # y = y[:, :, :4]  # if you want to reduce code size.
    z = model.decode(y)
torchaudio.save('output.wav', z, sr)
```

## sample audio

Audio references are sampled from LibriSpeech test-clean.

|Reference|SoundStream|
|---|---|
[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/6829_68769_6829-68769-0006.in.wav)|[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/6829_68769_6829-68769-0006.out.wav)
[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/8230_279154_8230-279154-0000.in.wav)|[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/8230_279154_8230-279154-0000.out.wav)
[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/908_157963_908-157963-0028.in.wav)|[audio link](https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/908_157963_908-157963-0028.out.wav)
