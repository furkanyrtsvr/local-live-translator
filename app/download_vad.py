"""One-time, checksum-verified Silero ONNX download. Inference remains offline."""
import hashlib
from pathlib import Path
import tempfile
import urllib.request

REVISION = '1e261b036686cd0017d500ee96acd1c4ba572a9d'
SHA256 = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3'
URL = f'https://raw.githubusercontent.com/snakers4/silero-vad/{REVISION}/src/silero_vad/data/silero_vad.onnx'


def main():
    directory = Path(__file__).resolve().parent.parent / 'models' / 'vad'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / 'silero_vad.onnx'
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print('Silero VAD modeli hazır; SHA256 doğrulandı.')
        return
    # Existing models survive an interrupted or invalid download.
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.download', delete=False) as output:
        temporary = Path(output.name)
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                while block := response.read(1024*1024):
                    output.write(block)
        except Exception:
            output.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Silero VAD SHA256 uyuşmuyor; model değiştirilmedi')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print('Silero VAD indirildi; SHA256 doğrulandı.')


if __name__ == '__main__':
    main()
