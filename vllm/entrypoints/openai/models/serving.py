# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from asyncio import Lock
from collections import defaultdict
from http import HTTPStatus

import msgspec

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    PinLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
    UnpinLoRAAdapterRequest,
)
from vllm.exceptions import LoRAAdapterNotFoundError
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.utils.counter import AtomicCounter

logger = init_logger(__name__)


class OpenAIModelRegistry:
    """Read-only view of the loaded base models with no engine dependency.

    Suitable for CPU-only / render-only contexts that have no engine client
    and no LoRA support.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        base_model_paths: list[BaseModelPath],
    ) -> None:
        self.model_config = model_config
        self.base_model_paths = base_model_paths
        self.lora_requests: dict[str, LoRARequest] = {}

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        return self.base_model_paths[0].name

    def is_base_model(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        """Return an ErrorResponse if model_name is not served, else None."""
        if not model_name or self.is_base_model(model_name):
            return None
        return create_error_response(
            message=f"The model `{model_name}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self.base_model_paths
            ]
        )

    async def resolve_lora(self, lora_name: str):
        raise RuntimeError("The OpenAIModelRegistry has no LoRA support.")


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    - /v1/pin_lora_adapter
    - /v1/unpin_lora_adapter
    """

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ):
        super().__init__()

        self.registry = OpenAIModelRegistry(
            model_config=engine_client.model_config,
            base_model_paths=base_model_paths,
        )

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self._pinned_lora_ids: set[int] = set()
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers():
            self.lora_resolvers.append(
                LoRAResolverRegistry.get_resolver(lora_resolver_name)
            )
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)

        self.model_config = self.engine_client.model_config
        self.renderer = self.engine_client.renderer
        self.input_processor = self.engine_client.input_processor

    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            load_request = LoadLoRAAdapterRequest(
                lora_path=lora.path,
                lora_name=lora.name,
                is_3d_lora_weight=lora.is_3d_lora_weight,
            )
            load_result = await self.load_lora_adapter(
                request=load_request, base_model_name=lora.base_model_name
            )
            if isinstance(load_result, ErrorResponse):
                raise ValueError(load_result.error.message)

    def is_base_model(self, model_name: str) -> bool:
        return self.registry.is_base_model(model_name)

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all
        adapters."""
        model_list = await self.registry.show_available_models()
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.path,
                parent=lora.base_model_name
                if lora.base_model_name
                else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_list.data.extend(lora_cards)
        return model_list

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
            previous = self.lora_requests.get(lora_name)
            lora_int_id = (
                previous.lora_int_id
                if previous is not None
                else self.lora_id_counter.inc(1)
            )
            was_pinned = previous is not None and lora_int_id in self._pinned_lora_ids
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
                load_inplace=request.load_inplace,
                is_3d_lora_weight=request.is_3d_lora_weight,
            )
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                if str(
                    LoRAAdapterNotFoundError(
                        lora_request.lora_name, lora_request.lora_path
                    )
                ) in str(e):
                    return create_error_response(
                        LoRAAdapterNotFoundError(
                            lora_request.lora_name, lora_request.lora_path
                        )
                    )
                return create_error_response(
                    message=str(e),
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            if previous is not None:
                # An in-place reload removes and re-adds the adapter in the
                # worker cache, clearing any existing pin on lora_int_id.
                self._pinned_lora_ids.discard(lora_int_id)

            if request.pinned:
                try:
                    await self.engine_client.pin_lora(lora_int_id)
                except Exception as e:
                    # Avoid leaving a brand-new adapter registered in the engine
                    # when the caller requested an atomic load+pin and pinning fails.
                    if previous is None:
                        try:
                            await self.engine_client.remove_lora(lora_int_id)
                        except Exception:
                            logger.warning(
                                "Failed to clean up LoRA adapter '%s' after pin error",
                                lora_name,
                                exc_info=True,
                            )
                    else:
                        # An in-place reload already replaced the worker's
                        # cached adapter under lora_int_id. Restore the previous
                        # version so frontend and worker stay on the same
                        # version; if rollback fails, advance lora_requests to
                        # the newly loaded version instead of diverging.
                        rollback_request = msgspec.structs.replace(
                            previous, load_inplace=True
                        )
                        try:
                            await self.engine_client.add_lora(rollback_request)
                        except Exception:
                            logger.warning(
                                "Failed to roll back LoRA adapter '%s' to '%s' "
                                "after pin error; keeping newly loaded path '%s' "
                                "in sync with the engine.",
                                lora_name,
                                previous.lora_path,
                                lora_path,
                                exc_info=True,
                            )
                            lora_request.load_inplace = False
                            self.lora_requests[lora_name] = lora_request
                        else:
                            if was_pinned:
                                try:
                                    await self.engine_client.pin_lora(lora_int_id)
                                except Exception:
                                    logger.warning(
                                        "Failed to re-pin rolled-back LoRA "
                                        "adapter '%s'",
                                        lora_name,
                                        exc_info=True,
                                    )
                                else:
                                    self._pinned_lora_ids.add(lora_int_id)
                    return self._lora_pin_error_response(e)
                self._pinned_lora_ids.add(lora_int_id)

            if previous is not None and previous.lora_path == lora_path:
                logger.warning(
                    "Reloaded LoRA adapter '%s' in place from the same path '%s'. "
                    "Prefix-cache blocks computed with its previous weights are "
                    "not invalidated; load changed weights from a new path to "
                    "avoid reusing them.",
                    lora_name,
                    lora_path,
                )
            # Reset load_inplace before caching so the flag does not leak to
            # subsequent generate requests.
            lora_request.load_inplace = False
            self.lora_requests[lora_name] = lora_request
            logger.info(
                "Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path
            )
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def pin_lora_adapter(
        self, request: PinLoRAAdapterRequest
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = self._check_existing_lora_adapter(lora_name, "pin")
            if error_check_ret is not None:
                return error_check_ret

            lora_request = self.lora_requests[lora_name]
            try:
                await self.engine_client.pin_lora(lora_request.lora_int_id)
            except Exception as e:
                return self._lora_pin_error_response(e)

            self._pinned_lora_ids.add(lora_request.lora_int_id)
            logger.info("Pinned LoRA adapter: name '%s'", lora_name)
            return f"Success: LoRA adapter '{lora_name}' pinned successfully."

    async def unpin_lora_adapter(
        self, request: UnpinLoRAAdapterRequest
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = self._check_existing_lora_adapter(lora_name, "unpin")
            if error_check_ret is not None:
                return error_check_ret

            lora_request = self.lora_requests[lora_name]
            try:
                await self.engine_client.unpin_lora(lora_request.lora_int_id)
            except Exception as e:
                return self._lora_pin_error_response(e)

            self._pinned_lora_ids.discard(lora_request.lora_int_id)
            logger.info("Unpinned LoRA adapter: name '%s'", lora_name)
            return f"Success: LoRA adapter '{lora_name}' unpinned successfully."

    def _check_existing_lora_adapter(
        self, lora_name: str, action: str
    ) -> ErrorResponse | None:
        if not lora_name:
            return create_error_response(
                message=f"'lora_name' needs to be provided to {action} a LoRA adapter.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if lora_name not in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{lora_name}' cannot be found.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return None

    @staticmethod
    def _lora_pin_error_response(exc: Exception) -> ErrorResponse:
        msg = str(exc)
        # AsyncMPClient wraps worker exceptions as plain Exception with the
        # original message appended, so check known user-error messages as well.
        user_error_markers = (
            "All items are pinned",
            "is not supported in LoRAModelManager",
            "is not registered",
            "not in cache",
            "LoRA is not enabled",
        )
        if isinstance(exc, (NotImplementedError, ValueError)) or any(
            marker in msg for marker in user_error_markers
        ):
            return create_error_response(
                message=msg,
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        return create_error_response(
            message=msg,
            err_type="InternalServerError",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    async def unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_unload_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            # Unpin if pinned so the engine's LRU cache does not permanently leak
            # a GPU slot after the adapter is removed from lora_requests.
            lora_int_id = self.lora_requests[lora_name].lora_int_id
            if lora_int_id in self._pinned_lora_ids:
                try:
                    await self.engine_client.unpin_lora(lora_int_id)
                except Exception:
                    logger.warning(
                        "Failed to unpin LoRA adapter '%s' during unload",
                        lora_name,
                        exc_info=True,
                    )
                self._pinned_lora_ids.discard(lora_int_id)

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

        if self.is_base_model(request.lora_name):
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' conflicts with a "
                "served base model.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # If not loading inplace
        # Check if the lora adapter with the given name already exists
        if not request.load_inplace and request.lora_name in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' has already been "
                "loaded. If you want to load the adapter in place, set 'load_inplace'"
                " to True.",
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
