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
        # Dev headers for nvdiffrast's JIT-compiled GL plugin
        "libegl1-mesa-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
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
            "CPATH": "/usr/local/cuda/include",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        }
    )
    .add_local_dir(
        "./submodules",
        remote_path=f"{REMOTE_REPO}/submodules",
        copy=True,
        ignore=[".git/**", "**/__pycache__/**", "**/*.pyc"],
    )
    .run_commands(
        # Patch nvdiffrast ops.py: capture cpp_extension.load()'s return value
        # instead of relying on importlib.import_module (defensive; idempotent).
        (
            f"sed -i 's|^\\(\\s*\\)torch\\.utils\\.cpp_extension\\.load(name=plugin_name|"
            f"\\1_cached_plugin[gl] = torch.utils.cpp_extension.load(name=plugin_name|' "
            f"{REMOTE_REPO}/submodules/nvdiffrast/nvdiffrast/torch/ops.py && "
            f"sed -i '/_cached_plugin\\[gl\\] = importlib\\.import_module(plugin_name)/d' "
            f"{REMOTE_REPO}/submodules/nvdiffrast/nvdiffrast/torch/ops.py"
        ),
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
        f"cd {REMOTE_REPO}/submodules/TopologyLayer && pip install --no-build-isolation ."
    )
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
    gpu="L40S",
    cpu=8.0, 
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
    gpu="L40S",
    cpu=8.0, 
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

# ============================================================================
# Sweep experiment: topology loss vs. Gaussian budget
# ============================================================================

@app.function(
    image=image,
    gpu="L40S",
    cpu=8.0,
    memory=32768,
    volumes=VOLUMES,
    timeout=6 * 60 * 60,  # 2 hr per cell (lowres should be 15-25 min)
)
def train_dtu_sweep_cell(
    scene: str,
    run_name: str,
    sampling_factor: float = 1.0,
    use_topo_loss: bool = False,
    topo_weight: float = 0.005,
    mesh_config: str = "lowres",
) -> dict:
    """Train + extract + collect metrics for one sweep cell."""
    output_dir = f"./output/sweep/{scene}/{run_name}"

    # --- Train ---
    train_cmd = (
        f"python train.py "                                  # was: train_regular_densification.py
        f"-s ./data/{scene} -m {output_dir} -r 2 "
        f"--rasterizer radegs --imp_metric indoor "
        f"--mesh_config {mesh_config} --decoupled_appearance "
        f"--sampling_factor {sampling_factor} "
        f"--log_interval 500"
    )
    if use_topo_loss:
        train_cmd += f" --use_topo_loss --topo_weight {topo_weight}"
    _run(train_cmd)

    # --- Extract mesh ---
    _run(
        f"python mesh_extract_sdf.py "
        f"-s ./data/{scene} -m {output_dir} "
        f"--iteration 18000 --refine_iter 0 "
        f"--rasterizer radegs --config {mesh_config} --imp_metric indoor "
        f"--remove_oof_vertices"
    )

    # --- Collect metrics ---
    metrics = _collect_sweep_metrics(f"{REMOTE_REPO}/milo/{output_dir.lstrip('./')}")
    metrics["run_name"] = run_name
    metrics["sampling_factor"] = sampling_factor
    metrics["use_topo_loss"] = use_topo_loss
    metrics["topo_weight"] = topo_weight if use_topo_loss else 0.0

    output_volume.commit()
    return metrics


def _collect_sweep_metrics(output_dir: str) -> dict:
    """Read final checkpoint + mesh and compute headline metrics."""
    from pathlib import Path
    import trimesh
    from plyfile import PlyData

    out = Path(output_dir)
    metrics = {"output_dir": str(out)}

    # Gaussian count from the final point cloud
    pc_dirs = sorted(
        (out / "point_cloud").glob("iteration_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if pc_dirs:
        final_pc = pc_dirs[-1] / "point_cloud.ply"
        if final_pc.exists():
            ply = PlyData.read(str(final_pc))
            metrics["gaussian_count"] = len(ply["vertex"].data)
        else:
            metrics["gaussian_count"] = None
    else:
        metrics["gaussian_count"] = None

    # Mesh stats — pick the extracted mesh
    mesh_candidates = list(out.glob("mesh_*.ply"))
    if mesh_candidates:
        # Prefer the post-processed one if it exists, else the first match
        mesh_path = next(
            (m for m in mesh_candidates if "post" in m.name.lower()),
            mesh_candidates[0],
        )
        mesh = trimesh.load(str(mesh_path), process=False)
        metrics["mesh_path"] = str(mesh_path)
        metrics["n_vertices"] = int(mesh.vertices.shape[0])
        metrics["n_faces"] = int(mesh.faces.shape[0])
        metrics["mesh_size_mb"] = mesh_path.stat().st_size / (1024 * 1024)

        # Connected components — the headline metric for CCLoss
        components = mesh.split(only_watertight=False)
        metrics["n_components"] = len(components)
        # Floater fraction: components much smaller than the largest
        if components:
            sizes = sorted((c.vertices.shape[0] for c in components), reverse=True)
            metrics["largest_component_frac"] = sizes[0] / sum(sizes)
            metrics["n_components_over_100v"] = sum(s > 100 for s in sizes)
    else:
        metrics["n_vertices"] = None
        metrics["n_components"] = None

    return metrics


@app.local_entrypoint()
def sweep(
    scene: str = "scan65",
    parallel: bool = True,
    mesh_config: str = "lowres",
    topo_weight: float = 0.005,
) -> None:
    """
    Run the 2x4 sweep: sampling_factor x {no topo, topo} on one DTU scene.

    Usage:
        modal run modal_app.py::sweep --scene scan24
        modal run modal_app.py::sweep --scene scan24 --no-parallel  # serialize
    """
    import json

    sampling_factors = [0.1, 0.2, 0.4, 1.0]
    cells = []
    for sf in sampling_factors:
        for topo in [False, True]:
            tag = "topo" if topo else "notopo"
            cells.append(
                dict(
                    scene=scene,
                    run_name=f"sf{sf}_{tag}",
                    sampling_factor=sf,
                    use_topo_loss=topo,
                    topo_weight=topo_weight,
                    mesh_config=mesh_config,
                )
            )

    print(f"[sweep] {len(cells)} cells on scene={scene} mesh_config={mesh_config}")
    print(f"[sweep] mode: {'parallel' if parallel else 'sequential'}\n")

    if parallel:
        handles = [train_dtu_sweep_cell.spawn(**c) for c in cells]
        results = []
        for c, h in zip(cells, handles):
            try:
                results.append(h.get())
            except Exception as e:
                print(f"[sweep] cell {c['run_name']} FAILED: {e}")
                results.append({**c, "error": str(e)})
    else:
        results = []
        for c in cells:
            try:
                results.append(train_dtu_sweep_cell.remote(**c))
            except Exception as e:
                print(f"[sweep] cell {c['run_name']} FAILED: {e}")
                results.append({**c, "error": str(e)})

    # --- Save raw results ---
    Path = __import__("pathlib").Path
    out_dir = Path(f"./milo/output/sweep/{scene}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Print summary table ---
    _print_summary(results)


def _print_summary(results: list) -> None:
    """Print a markdown-ish table of the sweep results."""
    cols = ["run_name", "gaussian_count", "n_vertices", "n_components",
            "n_components_over_100v", "largest_component_frac", "mesh_size_mb"]
    widths = {c: max(len(c), 14) for c in cols}

    header = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print("\n" + header)
    print(sep)

    for r in sorted(results, key=lambda x: (x.get("sampling_factor", 0), x.get("use_topo_loss", False))):
        if "error" in r:
            print(f"{r.get('run_name','?').ljust(widths['run_name'])} | FAILED: {r['error']}")
            continue
        row = []
        for c in cols:
            v = r.get(c, "—")
            if isinstance(v, float):
                v = f"{v:.3f}" if v < 100 else f"{v:.1f}"
            row.append(str(v).ljust(widths[c]))
        print(" | ".join(row))
    print()

# Add temporarily at the bottom of modal_app.py:
@app.local_entrypoint()
def sweep_test() -> None:
    result = train_dtu_sweep_cell.remote(
        scene="scan65",
        run_name="test_one",
        sampling_factor=0.10,
        use_topo_loss=True,
        mesh_config="lowres",
        topo_weight= 0.005
    )
    print(result)

@app.function(
    image=image,
    gpu="L40S",
    cpu=4.0,
    memory=16384,
    volumes=VOLUMES,
    timeout=30 * 60,
)
def extract_sdf(
    scene: str,
    model_dir: str = "",
    iteration: int = 18000,
    refine_iter: int = 0,
    rasterizer: str = "radegs",
    config: str = "lowres",
    imp_metric: str = "indoor",
) -> None:
    """Extract a mesh from a trained checkpoint using learned SDF values.

    Pass --refine-iter 0 to skip the refinement loop (recommended when validating
    topology-loss effects, since the default 1000-iter refinement has no topo loss
    and can undo cleanup).

    Example:
        modal run modal_app.py::extract_sdf --scene scan24 --model-dir scan24_topo
        modal run modal_app.py::extract_sdf --scene scan24 --model-dir scan24_out --refine-iter 0
    """
    if not model_dir:
        model_dir = f"{scene}_out"
    output_dir = f"./output/{model_dir}"

    _run(
        f"python mesh_extract_sdf.py --remove_oof_vertices --use_topo_loss "
        f"-s ./data/{scene} -m {output_dir} "
        f"--iteration {iteration} "
        f"--refine_iter {refine_iter} "
        f"--rasterizer {rasterizer} "
        f"--config {config} "
        f"--imp_metric {imp_metric}"
    )
    output_volume.commit()
    print(f"[extract_sdf] mesh written to {output_dir}/mesh_learnable_sdf.ply")


@app.local_entrypoint()
def extract(
    scene: str = "scan65",
    model_dir: str = "",
    iteration: int = 18000,
    refine_iter: int = 1000,
    rasterizer: str = "radegs",
    config: str = "lowres",
    imp_metric: str = "indoor",
) -> None:
    """Local entrypoint wrapping extract_sdf."""
    extract_sdf.remote(
        scene=scene,
        model_dir=model_dir,
        iteration=iteration,
        refine_iter=refine_iter,
        rasterizer=rasterizer,
        config=config,
        imp_metric=imp_metric,
    )