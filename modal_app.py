"""Modal app for running MILo on remote GPUs.

Usage (from the repo root, with `modal` installed and authed):

    # 1. Upload a COLMAP scene from your local machine into the data volume
    modal run modal_app.py::upload_scene --local-path ./milo/data/Ignatius

    # 2. Train + extract a mesh
    modal run modal_app.py --scene Ignatius --imp-metric outdoor

    # 3. Pull the trained model / mesh back down
    modal volume get milo-outputs Ignatius ./milo/output/

The first run builds the image (slow: ~15-25 min while CUDA extensions compile).
Subsequent runs reuse the cached image.
"""

from pathlib import Path

import modal

APP_NAME = "milo"
REMOTE_REPO = "/root/MILo"
# Build for common datacenter GPUs so the cached image works across GPU types.
# 7.5=T4, 8.0=A100, 8.6=A10G/3090, 8.9=L4/L40S/4090, 9.0=H100
TORCH_CUDA_ARCH_LIST = "7.5;8.0;8.6;8.9;9.0"
CMAKE_CUDA_ARCHITECTURES = "75;80;86;89;90"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("milo-data", create_if_missing=True)
output_volume = modal.Volume.from_name("milo-outputs", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "wget",
        "cmake",
        "build-essential",
        "ninja-build",
        "libgmp-dev",
        "libmpfr-dev",
        "libcgal-dev",
        "libboost-all-dev",
        "libgl1",
        "libglib2.0-0",
        "libegl1",
    )
    .uv_pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .uv_pip_install(
        "open3d==0.19.0",
        "trimesh==4.6.8",
        "scikit-image==0.24.0",
        "opencv-python==4.11.0.86",
        "plyfile==1.1",
        "tqdm==4.67.1",
        "ninja",
        "setuptools",
        "wheel",
    )
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST,
            "FORCE_CUDA": "1",
            "CC": "gcc",
            "CXX": "g++",
            # Make cuda_runtime.h visible to C++ compilations (mirrors MILo's install.py).
            "CPATH": "/usr/local/cuda/include",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    # Bake submodules into the image so build-cache is only busted when *they*
    # change (essentially never after `git submodule update --init`).
    .add_local_dir(
        "./submodules",
        remote_path=f"{REMOTE_REPO}/submodules",
        copy=True,
        ignore=[".git/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .run_commands(
        f"cd {REMOTE_REPO} && pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms",
        f"cd {REMOTE_REPO} && pip install --no-build-isolation submodules/diff-gaussian-rasterization",
        f"cd {REMOTE_REPO} && pip install --no-build-isolation submodules/diff-gaussian-rasterization_gof",
        f"cd {REMOTE_REPO} && pip install --no-build-isolation submodules/simple-knn",
        f"cd {REMOTE_REPO} && pip install --no-build-isolation submodules/fused-ssim",
        (
            f"cd {REMOTE_REPO}/submodules/tetra_triangulation && "
            f"cmake . -DCMAKE_CUDA_ARCHITECTURES='{CMAKE_CUDA_ARCHITECTURES}' && "
            f"make -j && pip install --no-build-isolation -e ."
        ),
        f"cd {REMOTE_REPO}/submodules/nvdiffrast && pip install --no-build-isolation -e .",
        f"cd {REMOTE_REPO}/submodules/TopologyLayer && pip install --no-build-isolation -e .",
    )
    # Mount the rest of the repo at *runtime*. `copy=False` (default) means this
    # is not part of the image hash, so editing milo/*.py never triggers a rebuild.
    # Exclude submodules/ because the editable installs (nvdiffrast,
    # tetra_triangulation) need the *baked* paths, not the mounted ones.
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO,
        ignore=[
            "submodules/**",
            ".git/**",
            "milo/data/**",
            "milo/output/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "modal_app.py",
            "run_modal.sh",
        ],
    )
)

DATA_DIR = f"{REMOTE_REPO}/milo/data"
OUTPUT_DIR = f"{REMOTE_REPO}/milo/output"
VOLUMES = {DATA_DIR: data_volume, OUTPUT_DIR: output_volume}


def _run(cmd: str) -> None:
    import subprocess

    print(f"[modal] $ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=f"{REMOTE_REPO}/milo")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    timeout=6 * 60 * 60,
)
def train(
    scene: str,
    imp_metric: str = "outdoor",
    rasterizer: str = "radegs",
    extra_args: str = "",
) -> None:
    _run(
        f"python train.py -s ./data/{scene} -m ./output/{scene} "
        f"--imp_metric {imp_metric} --rasterizer {rasterizer} {extra_args}"
    )
    output_volume.commit()


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes=VOLUMES,
    timeout=2 * 60 * 60,
)
def extract_mesh(
    scene: str,
    rasterizer: str = "radegs",
    method: str = "sdf",
    extra_args: str = "",
) -> None:
    script = {
        "sdf": "mesh_extract_sdf.py",
        "integration": "mesh_extract_integration.py",
        "tsdf": "mesh_extract_regular_tsdf.py",
    }[method]
    _run(
        f"python {script} -s ./data/{scene} -m ./output/{scene} "
        f"--rasterizer {rasterizer} {extra_args}"
    )
    output_volume.commit()


@app.function(image=image, volumes={DATA_DIR: data_volume}, timeout=60 * 60)
def _upload_scene_remote(scene_name: str, files: list[tuple[str, bytes]]) -> None:
    base = Path(DATA_DIR) / scene_name
    for rel, blob in files:
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
    data_volume.commit()
    print(f"[modal] uploaded {len(files)} files to {base}")


@app.local_entrypoint()
def upload_scene(local_path: str) -> None:
    """Upload a local COLMAP scene directory to the data volume."""
    root = Path(local_path).expanduser().resolve()
    assert root.is_dir(), f"{root} is not a directory"
    files = [
        (str(p.relative_to(root)), p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    ]
    print(f"Uploading {len(files)} files from {root} as scene '{root.name}'...")
    _upload_scene_remote.remote(root.name, files)


@app.local_entrypoint()
def main(
    scene: str = "Ignatius",
    imp_metric: str = "outdoor",
    rasterizer: str = "radegs",
    extract: bool = True,
    method: str = "sdf",
    extra_args: str = "",
) -> None:
    train.remote(scene, imp_metric, rasterizer, extra_args)
    if extract:
        extract_mesh.remote(scene, rasterizer, method)
