"""Package trained code and chosen model artifacts into a tar.gz archive.

Examples:
    python scripts/package_submission.py --artifacts model_artifacts_fbcsp_lda --out outputs/fbcsp_lda_submission.tar.gz
    python scripts/package_submission.py --artifacts model_artifacts_riemann_lr --out outputs/riemann_lr_submission.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="model_artifacts", help="Artifact directory to package.")
    parser.add_argument("--out", default="outputs/smes_mi_submission.tar.gz")
    parser.add_argument("--workdir", default="outputs/submission_package")
    args = parser.parse_args()

    artifacts_dir = ROOT / args.artifacts
    if not (artifacts_dir / "artifact_config.json").exists():
        raise FileNotFoundError(f"{artifacts_dir / 'artifact_config.json'} not found. Run training first.")

    workdir = ROOT / args.workdir
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    copytree_clean(ROOT / "src", workdir / "src")
    copytree_clean(ROOT / "submission", workdir / "submission")
    copytree_clean(artifacts_dir, workdir / "model_artifacts")
    shutil.copyfile(ROOT / "requirements.txt", workdir / "requirements.txt")
    shutil.copyfile(ROOT / "README.md", workdir / "README.md")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with tarfile.open(out_path, "w:gz") as tar:
        for item in workdir.rglob("*"):
            tar.add(item, arcname=item.relative_to(workdir))

    print(f"Created submission archive: {out_path}")
    print(f"Packaged artifacts from: {artifacts_dir}")


if __name__ == "__main__":
    main()
