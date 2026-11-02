"""Bounded image decoding shared by the UI and local inference examples."""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def decode_image(contents: bytes) -> Image.Image:
    """Decode a bounded PNG/JPEG upload into an independent RGB preview."""
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Image upload exceeds the 20 MiB preview limit.")
    try:
        with Image.open(BytesIO(contents)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Upload a PNG or JPEG image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 16 megapixel preview limit.")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file could not be decoded as a valid image.") from exc
