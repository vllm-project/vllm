# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    LoadLoRAAdapterRequest,
    ModelCard,
    ModelList,
    ModelPermission,
    UnloadLoRAAdapterRequest,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.utils.counter import AtomicCounter

logger = init_logger(__name__)


@dataclass
class BaseModelPath:
    name: str
    model_path: str


@dataclass
class LoRAModulePath:
    name: str
    path: str
    base_model_name: str | None = None


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    """

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ):
        super().__init__()

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers():
            self.lora_resolvers.append(
                LoRAResolverRegistry.get_resolver(lora_resolver_name)
            )
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)

        self.input_processor = self.engine_client.input_processor
        self.io_processor = self.engine_client.io_processor
        self.renderer_config = self.engine_client.renderer_config
        self.model_config = self.engine_client.model_config
        self.max_model_len = self.model_config.max_model_len

    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            load_request = LoadLoRAAdapterRequest(
                lora_path=lora.path, lora_name=lora.name
            )
            load_result = await self.load_lora_adapter(
                request=load_request, base_model_name=lora.base_model_name
            )
            if isinstance(load_result, ErrorResponse):
                raise ValueError(load_result.error.message)

    def is_base_model(self, model_name) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        """Returns the appropriate model name depending on the availability
        and support of the LoRA or base model.
        Parameters:
        - lora: LoRARequest that contain a base_model_name.
        Returns:
        - str: The name of the base model or the first available model path.
        """
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all
        adapters"""
        model_cards = [
            ModelCard(
                id=base_model.name,
                max_model_len=self.max_model_len,
                root=base_model.model_path,
                permission=[ModelPermission()],
            )
            for base_model in self.base_model_paths
        ]
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.local_path,
                parent=lora.base_model_name
                if lora.base_model_name
                else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_load_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            lora_path = request.lora_path
            unique_id = self.lora_id_counter.inc(1)
            lora_request = LoRARequest(
                lora_name=lora_name, lora_int_id=unique_id, lora_path=lora_path
            )
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                error_type = "BadRequestError"
                status_code = HTTPStatus.BAD_REQUEST
                if "No adapter found" in str(e):
                    error_type = "NotFoundError"
                    status_code = HTTPStatus.NOT_FOUND

                return create_error_response(
                    message=str(e), err_type=error_type, status_code=status_code
                )

            self.lora_requests[lora_name] = lora_request
            logger.info(
                "Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path
            )
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_unload_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            # Safe to delete now since we hold the lock
            del self.lora_requests[lora_name]
            logger.info("Removed LoRA adapter: name '%s'", lora_name)
            return f"Success: LoRA adapter '{lora_name}' removed successfully."

    async def _check_load_lora_adapter_request(
        self, request: LoadLoRAAdapterRequest
    ) -> ErrorResponse | None:
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Check if the lora adapter with the given name already exists
        if request.lora_name in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' has already been "
                "loaded.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        return None

    async def _check_unload_lora_adapter_request(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | None:
        # Check if 'lora_name' is not provided return an error
        if not request.lora_name:
            return create_error_response(
                message="'lora_name' needs to be provided to unload a LoRA adapter.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Check if the lora adapter with the given name exists
        if request.lora_name not in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return None

    async def resolve_lora(self, lora_name: str) -> LoRARequest | ErrorResponse:
        """Attempt to resolve a LoRA adapter using available resolvers.

        Args:
            lora_name: Name/identifier of the LoRA adapter

        Returns:
            LoRARequest if found and loaded successfully.
            ErrorResponse (404) if no resolver finds the adapter.
            ErrorResponse (400) if adapter(s) are found but none load.
        """
        async with self.lora_resolver_lock[lora_name]:
            # First check if this LoRA is already loaded
            if lora_name in self.lora_requests:
                return self.lora_requests[lora_name]

            base_model_name = self.model_config.model
            unique_id = self.lora_id_counter.inc(1)
            found_adapter = False

            # Try to resolve using available resolvers
            for resolver in self.lora_resolvers:
                lora_request = await resolver.resolve_lora(base_model_name, lora_name)

                if lora_request is not None:
                    found_adapter = True
                    lora_request.lora_int_id = unique_id

                    try:
                        await self.engine_client.add_lora(lora_request)
                        self.lora_requests[lora_name] = lora_request
                        logger.info(
                            "Resolved and loaded LoRA adapter '%s' using %s",
                            lora_name,
                            resolver.__class__.__name__,
                        )
                        return lora_request
                    except BaseException as e:
                        logger.warning(
                            "Failed to load LoRA '%s' resolved by %s: %s. "
                            "Trying next resolver.",
                            lora_name,
                            resolver.__class__.__name__,
                            e,
                        )
                        continue

            if found_adapter:
                # An adapter was found, but all attempts to load it failed.
                return create_error_response(
                    message=(
                        f"LoRA adapter '{lora_name}' was found but could not be loaded."
                    ),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            else:
                # No adapter was found
                return create_error_response(
                    message=f"LoRA adapter {lora_name} does not exist",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorInfo(message=message, type=err_type, code=status_code.value)
    )
