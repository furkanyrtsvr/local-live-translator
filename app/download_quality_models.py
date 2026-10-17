"""Download larger local ASR/translation models, checking their published LFS hashes."""
import concurrent.futures
import hashlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ('ggerganov/whisper.cpp', 'ggml-small.bin', 'models/whisper',
     '5359861c739e955e79d9a303bcbc70fb988958b1'),
    ('mradermacher/translategemma-4b-it-GGUF', 'translategemma-4b-it.Q5_K_M.gguf', 'models/llm',
     'd37868f759d2f817a43d28dd80cf9f982ffa62fd'),
]


def download(spec):
    repo, name, directory, revision = spec
    session = requests.Session()
    session.trust_env = False
    metadata = session.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=30)
    metadata.raise_for_status()
    info = metadata.json()
    file = next(item for item in info['siblings'] if item['rfilename'] == name)
    expected = file['lfs']['sha256']
    destination = ROOT / directory / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest() == expected:
        print(f'{name}: already verified', flush=True)
        return
    temporary = destination.with_suffix(destination.suffix + '.download')
    digest = hashlib.sha256()
    received, announced = 0, 0
    with session.get(f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{name}',
                     stream=True, timeout=(20, 60)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for block in response.iter_content(1024*1024):
                output.write(block)
                digest.update(block)
                received += len(block)
                if received-announced >= 256*1024*1024:
                    print(f'{name}: {received/1024**2:.0f} MiB', flush=True)
                    announced = received
    if digest.hexdigest() != expected:
        raise ValueError(f'{name}: SHA256 mismatch; existing model preserved')
    temporary.replace(destination)
    print(f'{name}: verified {expected}, revision {info["sha"]}', flush=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, MODELS))
