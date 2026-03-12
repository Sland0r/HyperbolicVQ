import urllib.request
import tarfile
import os

DATASET_DIR = '/home/acolombo/VAEs/dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

splits = [
    ('train-clean-100', 'https://www.openslr.org/resources/60/train-clean-100.tar.gz'),
    ('dev-clean', 'https://www.openslr.org/resources/60/dev-clean.tar.gz'),
]

for name, url in splits:
    dest = os.path.join(DATASET_DIR, 'LibriTTS', name)
    if os.path.isdir(dest):
        print(f"Skipping {name} (already exists at {dest})")
        continue

    archive = os.path.join(DATASET_DIR, f'{name}.tar.gz')
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, archive)

    print(f"Extracting {name}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=DATASET_DIR)

    print(f"Removing archive for {name}.")
    os.remove(archive)

print("Done. All splits downloaded.")
