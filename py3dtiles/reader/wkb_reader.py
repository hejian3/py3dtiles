from pathlib import Path


def get_metadata(path: Path, color_scale=None) -> dict:
    return {
        'type': 'b3dm',
        'portions': [str(path)]
    }


def run(filename: str) -> bytes:
    with open(filename, 'rb') as f:
        return f.read()
