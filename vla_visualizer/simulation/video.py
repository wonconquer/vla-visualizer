"""Encode only frames from an actual policy-stepped rollout, without synthetic motion."""

from pathlib import Path
import subprocess


def encode_rollout(directory, record):
    frames = record.get("frames", [])
    if not record.get("simulator_integration_success") or len(frames) < 2:
        return None
    directory = Path(directory).resolve()
    lines = []
    for index, frame in enumerate(frames):
        path = (directory / frame["file"]).resolve()
        if not path.is_relative_to(directory) or path.suffix != ".png" or not path.is_file():
            raise ValueError("Rollout video contains an invalid frame reference.")
        # Generated frame names contain only safe path characters.
        relative = path.relative_to(directory).as_posix()
        if any(c in relative for c in "'\n\r\\"):
            raise ValueError("Unsupported frame filename.")
        lines.append(f"file '{relative}'")
        if index < len(frames) - 1:
            dt = frames[index + 1]["simulator_time"] - frame["simulator_time"]
            if not 0 < dt <= 5:
                raise ValueError("Invalid recorded simulator frame interval.")
            lines.append(f"duration {dt:.9f}")
    listing = directory / "video_frames.txt"
    listing.write_text("\n".join(lines) + "\n")
    staging = directory / ".policy_rollout.staging.mp4"
    final = directory / "policy_rollout.mp4"
    if final.exists():
        raise FileExistsError("An existing rollout video will not be overwritten.")
    try:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-threads",
                    "2",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-vsync",
                    "vfr",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    str(staging),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Rollout video encoding exceeded its 30-second limit.") from exc
        if result.returncode:
            raise RuntimeError("Rollout video encoding failed: " + result.stderr[-1000:])
        if staging.stat().st_size > 32 * 1024**2:
            raise ValueError("Rollout video exceeds its 32 MiB encoding limit.")
        staging.replace(final)
        return {
            "file": final.name,
            "bytes": final.stat().st_size,
            "source": "actual policy-stepped simulator frames",
            "timing": "variable frame durations from recorded simulator times",
            "playback_speed": 1.0,
            "available": "after completion only",
        }
    finally:
        staging.unlink(missing_ok=True)
