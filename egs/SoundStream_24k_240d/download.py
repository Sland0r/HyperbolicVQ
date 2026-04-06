import torchaudio

dataset = torchaudio.datasets.LIBRITTS(
    root="/scratch-shared/acolombo/",
    url="train-other-500",
    download=True
)
