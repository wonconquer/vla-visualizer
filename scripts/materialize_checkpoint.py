"""Explicitly export base plus delta as a standalone checkpoint."""

import argparse
from pathlib import Path
import shutil

from vla_visualizer.adapters.smolvla import CHECKPOINT_FILES
from vla_visualizer.execution.training import load_base
from vla_visualizer.execution.checkpoints import load_trained_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("The output directory already exists and will not be overwritten.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 2 * 1024**3:
        raise RuntimeError("Standalone float32 export requires about 1.80 GB plus temporary free space.")
    staging = args.output.with_name(args.output.name + ".staging")
    if staging.exists():
        raise ValueError("A staging directory already exists. It will not be deleted automatically.")
    adapter = None
    try:
        adapter = load_base({"device": "cpu", "seed": 0})
        load_trained_checkpoint(adapter, args.checkpoint)
        staging.mkdir()
        from safetensors.torch import save_file

        save_file(
            {n: t.detach().cpu().contiguous() for n, t in adapter._policy.state_dict().items()},
            str(staging / "model.safetensors"),
        )
        for name in CHECKPOINT_FILES:
            if name != "model.safetensors":
                shutil.copyfile(Path(adapter._snapshot_path) / name, staging / name)
        staging.rename(args.output)
        print(args.output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        if adapter is not None:
            adapter.unload()


if __name__ == "__main__":
    main()
