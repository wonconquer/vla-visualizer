# Third-party sources

The application uses pretrained models and recorded robot observations. It does
not claim the SmolVLA architecture, pretrained weights, or SO100 dataset as new work.

- [LeRobot](https://github.com/huggingface/lerobot): policy and processor implementation;
  version 0.6.1 is pinned in the environment.
- [SmolVLA](https://huggingface.co/lerobot/smolvla_base): upstream model family.
- [SO100 fine-tuned checkpoint](https://huggingface.co/Zekai-Chen/smolvla_so100_pickplace_ft_2cam):
  policy revision `77172eaffe82c0897945607bfe818b5b164eb57e`. Weights are downloaded separately.
- [SO100 pick-and-place dataset](https://huggingface.co/datasets/lerobot/svla_so100_pickplace/blob/87169e502b742f989b96c952bff7af55e248d467/README.md):
  the pinned dataset metadata identifies Apache-2.0 as its license. The included
  two images and episode-0 Parquet come from this revision. The license text is
  included in [third_party/Apache-2.0.txt](third_party/Apache-2.0.txt).
- [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct):
  tokenizer/configuration revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

PyTorch, Transformers, Streamlit, Graphviz, and other dependencies are installed
from the pinned requirements rather than vendored here. Their licenses remain
with their respective distributions. No third-party model weights are committed.
