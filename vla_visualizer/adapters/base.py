"""Application-owned adapter API, not an assumed LeRobot API."""

from abc import ABC, abstractmethod

from vla_visualizer.contracts import (
    CaptureOptions,
    InferenceResult,
    InputSpec,
    LoadOptions,
    ModelCapabilities,
    ModelStructure,
    Observation,
    PreparedInputs,
)


class BaseVLAAdapter(ABC):
    """Isolate model-specific loading, preprocessing, inference, and inspection."""

    def get_inspection_spec(self) -> dict:
        """Return source-verified presentation relationships; never execute a model."""
        return {}

    @abstractmethod
    def load(self, options: LoadOptions) -> None:
        """Load lazily, freeze parameters, and select evaluation mode."""

    @abstractmethod
    def unload(self) -> None:
        """Release model references, hooks, and device resources."""

    @abstractmethod
    def get_input_spec(self) -> InputSpec:
        """Return requirements verified against the selected implementation."""

    @abstractmethod
    def get_capabilities(self) -> ModelCapabilities:
        """Report only supported and verified inspection features."""

    @abstractmethod
    def get_structure(self) -> ModelStructure:
        """Describe stages and their verified module paths."""

    @abstractmethod
    def preprocess(self, observation: Observation) -> PreparedInputs:
        """Validate and convert observations to the model's native inputs."""

    @abstractmethod
    def run_inference(self, inputs: PreparedInputs, capture: CaptureOptions) -> InferenceResult:
        """Run frozen inference and return actions and captures from the same run."""
