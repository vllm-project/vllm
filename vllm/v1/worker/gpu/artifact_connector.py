# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.artifact_transfer import (
    artifact_transfer_state,
    get_artifact_transfer_group,
    has_artifact_transfer_group,
)
from vllm.distributed.artifact_transfer.artifact_connector.v1.base import (
    ArtifactConnectorOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import LogprobsLists, ModelRunnerOutput


class ArtifactConnector:
    """ArtifactConnector interface used by GPUModelRunner."""

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        pass

    def record_model_runner_output(
        self,
        model_runner_output: "ModelRunnerOutput",
        finished_req_ids: set[str] | None = None,
    ) -> None:
        pass

    def post_forward(
        self, finished_req_ids: set[str]
    ) -> ArtifactConnectorOutput | None:
        return None

    def set_disabled(self, disabled: bool) -> None:
        pass


class ActiveArtifactConnector(ArtifactConnector):
    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.artifact_connector = get_artifact_transfer_group()
        self._disabled = False

    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        artifact_connector_metadata = scheduler_output.artifact_connector_metadata
        if artifact_connector_metadata is None:
            return
        self.artifact_connector.bind_connector_metadata(artifact_connector_metadata)
        self.artifact_connector.start_artifact_transfers()

    def _selected_request_ids(
        self,
        model_runner_output: "ModelRunnerOutput",
        finished_req_ids: set[str] | None,
    ) -> list[str]:
        req_ids = list(model_runner_output.req_ids)
        if (
            self.vllm_config.artifact_transfer_config is not None
            and self.vllm_config.artifact_transfer_config.transfer_mode == "final"
        ):
            finished_req_ids = finished_req_ids or set()
            req_ids = [req_id for req_id in req_ids if req_id in finished_req_ids]
        return req_ids

    @staticmethod
    def _serialize_logprobs(
        logprobs: "LogprobsLists", req_idx: int, num_positions: int
    ) -> dict[str, Any]:
        request_logprobs = logprobs.slice_request(req_idx, num_positions)
        return {
            "logprob_token_ids": request_logprobs.logprob_token_ids.tolist(),
            "logprobs": request_logprobs.logprobs.tolist(),
            "sampled_token_ranks": request_logprobs.sampled_token_ranks.tolist(),
        }

    def _build_artifact_payload(
        self,
        model_runner_output: "ModelRunnerOutput",
        request_id: str,
    ) -> dict[str, Any] | None:
        artifact_transfer_config = self.vllm_config.artifact_transfer_config
        if artifact_transfer_config is None:
            return None

        req_idx = model_runner_output.req_id_to_index[request_id]
        export_fields = list(artifact_transfer_config.export_fields)
        requested_fields = set(export_fields or ["token_ids"])
        payload: dict[str, Any] = {"request_id": request_id}

        token_ids: list[int] | None = None
        if req_idx < len(model_runner_output.sampled_token_ids):
            token_ids = model_runner_output.sampled_token_ids[req_idx]

        if token_ids is not None and "token_ids" in requested_fields:
            payload["token_ids"] = token_ids
        if token_ids is not None and "sampled_token_ids" in requested_fields:
            payload["sampled_token_ids"] = token_ids

        if (
            "logprobs" in requested_fields
            and token_ids is not None
            and model_runner_output.logprobs is not None
        ):
            payload["logprobs"] = self._serialize_logprobs(
                model_runner_output.logprobs, req_idx, len(token_ids)
            )

        if (
            "num_nans_in_logits" in requested_fields
            and model_runner_output.num_nans_in_logits is not None
            and request_id in model_runner_output.num_nans_in_logits
        ):
            payload["num_nans_in_logits"] = model_runner_output.num_nans_in_logits[
                request_id
            ]

        return payload if len(payload) > 1 else None

    def record_model_runner_output(
        self,
        model_runner_output: "ModelRunnerOutput",
        finished_req_ids: set[str] | None = None,
    ) -> None:
        if self._disabled:
            return
        if not self.artifact_connector.has_connector_metadata():
            return

        request_ids = self._selected_request_ids(model_runner_output, finished_req_ids)
        artifacts: dict[str, Any] = {}
        for request_id in request_ids:
            artifact_payload = self._build_artifact_payload(
                model_runner_output, request_id
            )
            if artifact_payload is not None:
                artifacts[request_id] = artifact_payload

        if artifacts:
            self.artifact_connector.record_step_artifacts(
                request_ids=request_ids, artifacts=artifacts
            )

    def post_forward(
        self, finished_req_ids: set[str]
    ) -> ArtifactConnectorOutput | None:
        if self._disabled:
            return None

        output = self.artifact_connector.get_artifact_connector_output()
        if output is None:
            worker_meta = self.artifact_connector.build_connector_worker_meta()
            finished_sending = self.artifact_connector.get_finished(finished_req_ids)
            if worker_meta is not None or finished_sending:
                output = ArtifactConnectorOutput(
                    finished_sending=finished_sending,
                    worker_meta=worker_meta,
                )
        self.artifact_connector.clear_connector_metadata()
        return output

    def set_disabled(self, disabled: bool) -> None:
        artifact_transfer_state._ARTIFACT_CONNECTOR_AGENT = (
            None if disabled else self.artifact_connector
        )
        self._disabled = disabled


NO_OP_ARTIFACT_CONNECTOR = ArtifactConnector()


def get_artifact_connector(vllm_config: VllmConfig) -> ArtifactConnector:
    if not has_artifact_transfer_group():
        return NO_OP_ARTIFACT_CONNECTOR
    return ActiveArtifactConnector(vllm_config)
