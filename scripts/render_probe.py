"""Test actual EGL and OSMesa rendering in isolated subprocesses."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = '''
import json
import mujoco
import numpy as np
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/><geom type="sphere" size=".1" rgba="1 0 0 1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, height=64, width=64) as renderer:
    renderer.update_scene(data)
    frame = renderer.render()
    assert frame.shape == (64,64,3) and frame.dtype == np.uint8 and frame.max() > frame.min()
print(json.dumps({'rendered':True,'shape':list(frame.shape),'mujoco':mujoco.__version__}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/jobs_validation/render_probe.json"))
    args = parser.parse_args()
    records = []
    for backend in ("egl", "osmesa"):
        try:
            result = subprocess.run([sys.executable, "-c", CODE],
                env={**os.environ, "MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend},
                capture_output=True, text=True, timeout=30)
            records.append({"backend": backend, "returncode": result.returncode,
                            "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        except subprocess.TimeoutExpired:
            records.append({"backend": backend, "error": "Rendering exceeded the 30-second limit"})
    selected = next((r["backend"] for r in records if r.get("returncode") == 0), None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"selected_backend": selected, "results": records}, ensure_ascii=False, indent=2))
    print(args.output)
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
