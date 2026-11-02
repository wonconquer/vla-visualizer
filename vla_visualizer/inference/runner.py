"""Serialize model lifecycle and inference; retain at most one loaded policy."""

from threading import RLock
from time import perf_counter

from vla_visualizer.adapters import registry
from vla_visualizer.adapters.base import BaseVLAAdapter
from vla_visualizer.contracts import (
    CaptureOptions,
    InferenceResult,
    InputSpec,
    LoadOptions,
    ModelSpec,
    ModelStructure,
    Observation,
)


class ModelUnavailableError(RuntimeError):
    """A requested model is not loaded in this process."""


class InferenceRunner:
    """A process-wide Streamlit resource; model operations hold the same lock."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._adapter: BaseVLAAdapter | None = None
        self._model_id: str | None = None
        self._lease_context = None

    def list_models(self) -> tuple[ModelSpec, ...]:
        return registry.list_models()

    def get_input_spec(self, model_id: str) -> InputSpec:
        return registry.create_adapter(model_id).get_input_spec()

    def get_structure(self, model_id: str) -> ModelStructure:
        return registry.create_adapter(model_id).get_structure()

    def get_inspection_spec(self, model_id: str) -> dict:
        return registry.create_adapter(model_id).get_inspection_spec()

    def is_loaded(self, model_id: str) -> bool:
        with self._lock:
            return self._adapter is not None and self._model_id == model_id

    def load(self, model_id: str, options: LoadOptions) -> None:
        with self._lock:
            self.unload()
            from vla_visualizer.execution.jobs import model_lease

            lease = model_lease()
            lease.__enter__()
            try:
                adapter = registry.create_adapter(model_id)
            except Exception:
                lease.__exit__(None, None, None)
                raise
            try:
                adapter.load(options)
            except Exception:
                adapter.unload()
                lease.__exit__(None, None, None)
                raise
            self._adapter, self._model_id = adapter, model_id
            self._lease_context = lease

    def unload(self) -> None:
        with self._lock:
            adapter, self._adapter = self._adapter, None
            self._model_id = None
            if adapter is not None:
                try:
                    adapter.unload()
                finally:
                    if self._lease_context is not None:
                        self._lease_context.__exit__(None, None, None)
                        self._lease_context = None

    def run_inference(
        self, model_id: str, observation: Observation, capture: CaptureOptions | None = None
    ) -> InferenceResult:
        registry.get_model(model_id)
        with self._lock:
            if not self.is_loaded(model_id):
                raise ModelUnavailableError("Load the model before running inference.")
            start = perf_counter()
            try:
                prepared = self._adapter.preprocess(observation)
                result = self._adapter.run_inference(prepared, capture or CaptureOptions())
            except Exception as exc:
                # Validation failures leave a usable policy loaded; execution failures
                # release policy caches and require an explicit clean reload.
                if not isinstance(exc, (ValueError, NotImplementedError)):
                    self.unload()
                raise
            return InferenceResult(
                actions=result.actions,
                intermediates=result.intermediates,
                metadata={**result.metadata, "pipeline_seconds": perf_counter() - start},
            )
