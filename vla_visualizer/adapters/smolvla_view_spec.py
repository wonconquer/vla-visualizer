"""Source-verified relationships for the approved SmolVLA pin, not measured timing."""

from vla_visualizer.adapters.smolvla_config import SmolVLACheckpoint
from vla_visualizer.adapters.smolvla_inspection import POINTS


def inspection_spec(checkpoint: SmolVLACheckpoint) -> dict:
    """Keep model semantics in the adapter; generic viewers consume descriptors."""
    cameras = [source for source, _ in checkpoint.camera_map]
    stages = []
    for index, camera in enumerate(cameras):
        stages.append(
            {
                "id": f"camera_{index}",
                "label": camera,
                "kind": "vision",
                "camera": camera,
                "records": ["camera_preprocessing", "vision", "connector"],
                "explanation": "Native RGB preprocessing → contextual vision tokens → connector tokens.",
            }
        )
    stages += [
        {
            "id": "language",
            "label": "Language",
            "kind": "language",
            "records": ["language_tokenization", "language"],
            "explanation": "Newline-formatted instruction → token IDs/mask → learned embeddings.",
        },
        {
            "id": "state",
            "label": "Robot state",
            "kind": "state",
            "records": ["robot_state_normalization", "state"],
            "explanation": "Measured state → batch/normalize → zero-pad → learned state token.",
        },
        {
            "id": "prefix",
            "label": "Multimodal prefix / cached context",
            "kind": "prefix",
            "records": ["multimodal_prefix"],
            "explanation": "Scaled camera/language embeddings and one state token; prefix computed once, KV context reused by expert.",
        },
        {
            "id": "generation",
            "label": "Frozen action generation",
            "kind": "flow",
            "records": [
                "initial_noise",
                "flow_state",
                "time_conditioned_actions",
                "action_velocity",
            ],
            "explanation": "Seeded noise → action/time conditioning → context-conditioned expert → velocity; native Euler integration.",
        },
        {
            "id": "actions",
            "label": "Predicted action chunk",
            "kind": "actions",
            "records": ["normalized_action_chunk", "action_postprocessing"],
            "explanation": "Native integrated output sliced to robot dimensions, then inverse-normalized absolute targets.",
        },
    ]
    return {
        "spec_version": 1,
        "checkpoint": checkpoint.repo_id,
        "camera_to_native": dict(checkpoint.camera_map),
        "camera_invocation_order": {"vision": cameras, "connector": cameras},
        "revision": checkpoint.revision,
        "record_paths": {
            **POINTS,
            "robot_state_normalization": "policy_preprocessor",
            "camera_preprocessing": "policy_preprocessor",
            "language_tokenization": "policy_preprocessor",
            "initial_noise": "SmolVLAPolicy.predict_action_chunk(noise=)",
            "normalized_action_chunk": "SmolVLAPolicy.predict_action_chunk",
            "action_postprocessing": "policy_postprocessor",
        },
        "output_ranks": {
            **dict.fromkeys(POINTS, 3),
            "state": 2,
            "robot_state_normalization": 2,
            "camera_preprocessing": 4,
            "language_tokenization": 2,
            "initial_noise": 3,
            "normalized_action_chunk": 3,
            "action_postprocessing": 3,
        },
        "dataset": checkpoint.dataset,
        "dataset_revision": checkpoint.dataset_revision,
        "stages": stages,
        "edges": [[f"camera_{i}", "prefix", "connector × sqrt(width)"] for i in range(len(cameras))]
        + [
            ["language", "prefix", "embedding × sqrt(width)"],
            ["state", "prefix", "state token"],
            ["prefix", "generation", "cached prefix context (computed once)"],
            ["generation", "actions", "Euler integration → native slice → inverse normalization"],
        ],
        "prefix_sources": [
            *[{"stage": "connector", "camera": camera, "label": camera} for camera in cameras],
            {"stage": "language", "label": "language (includes masked padding)"},
            {"stage": "state", "label": "one projected state token"},
        ],
        "prefix_special_tokens": False,
        "prefix_min_length": 0,
        "verified_special_tokens": {"2": "<|im_end|>"},
        "special_token_id_ranges": [[0, 16], [49152, 49279]],
        "special_token_evidence": "Pinned SmolVLM2 tokenizer_config.json added_tokens_decoder special=True IDs exhaustively verified as 0..16 and 49152..49279; ID 2=<|im_end|> is also the pad token.",
        "mask_explanation": "Native prefix padding mask: present camera tokens + actual language mask + state=True. Attention block boundaries keep image/language from attending to state/actions; expert uses prefix KV context and suffix mask. Masks are not attention probabilities.",
        "state_pipeline": "Batch axis is added before native normalization; raw batched values were not separately captured. State projection input contains native right zero-padding.",
        "joint_names": list(checkpoint.joint_names),
        "joint_units": ["legacy calibrated degrees"] * 5 + ["calibrated percent (nominal 0–100)"],
        "coordinate_convention": checkpoint.coordinate_convention,
        "action_semantics": "absolute target joint positions; predicted, not executed",
        "spatial_evidence": "LeRobot 0.6.1 embed_image + Transformers 5.4.0 SmolVLMVisionEmbeddings.flatten(2).transpose(1,2), no CLS; connector.pixel_shuffle groups adjacent rows/columns. Geometry recorded at actual vision input. Index charts do not assert attribution.",
        "time_semantics": "schedule_time is derived from native Euler schedule, not a recorded time tensor or physical robot time",
        "summary_method": "Float32 reductions of detached actual tensors; population std, L2 norm; bounded previews and indexed last-axis norms. No full large activations.",
    }
