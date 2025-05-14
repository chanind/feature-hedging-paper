from pathlib import Path


def glob_matches(base_dir: Path, glob_pattern: str) -> bool:
    return len(list(base_dir.glob(glob_pattern))) > 0
