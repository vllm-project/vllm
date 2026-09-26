# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SynthID-Text watermark generation primitives."""

import torch

from vllm.v1.watermarking.watermarker import (
    RandomSampler,
    Watermarker,
    WatermarkSample,
)


class SynthIDWatermarker(Watermarker):
    def __init__(
        self,
        ngram_len: int,
        keys: list[int],
        sampling_table_size: int,
        sampling_table_seed: int,
    ) -> None:
        if ngram_len < 2:
            raise ValueError("ngram_len must be at least 2")
        if not keys:
            raise ValueError("keys must not be empty")
        if sampling_table_size <= 0:
            raise ValueError("sampling_table_size must be positive")

        self.ngram_len = ngram_len
        self._context_width = ngram_len - 1
        self._keys = keys
        self.sampling_table_size = sampling_table_size
        self.sampling_table_seed = sampling_table_seed

        self._sampling_tables: dict[torch.device, torch.Tensor] = {}
        self._keys_by_device: dict[torch.device, torch.Tensor] = {}

    @property
    def context_width(self) -> int:
        return self._context_width

    def _get_keys(self, device: torch.device) -> torch.Tensor:
        keys = self._keys_by_device.get(device)
        if keys is None:
            keys = torch.tensor(self._keys, dtype=torch.long, device=device)
            self._keys_by_device[device] = keys
        return keys

    def _get_sampling_table(self, device: torch.device) -> torch.Tensor:
        table = self._sampling_tables.get(device)
        if table is None:
            generator = torch.Generator(device=device).manual_seed(
                self.sampling_table_seed
            )
            table = torch.randint(
                low=0,
                high=2,
                size=(self.sampling_table_size,),
                generator=generator,
                device=device,
            )
            self._sampling_tables[device] = table
        return table

    @staticmethod
    def _accumulate_hash(
        current_hash: torch.Tensor,
        data: torch.Tensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.Tensor:
        for i in range(data.shape[-1]):
            current_hash = torch.add(current_hash, data[..., i])
            current_hash = torch.mul(current_hash, multiplier)
            current_hash = torch.add(current_hash, increment)
        return current_hash

    def _compute_keys(
        self,
        contexts: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a SynthID key for every candidate token and watermark depth.

        Args:
            contexts:
                Previous ``ngram_len - 1`` token IDs for each request.
                Shape: [batch_size, context_width].

            indices:
                Candidate token IDs for each request.
                Usually the whole vocabulary.
                Shape: [batch_size, vocab_size].

        Returns:
            Integer SynthID keys with shape
            [batch_size, vocab_size, depth], where ``depth == len(keys)``.

        """
        batch_size = contexts.shape[0]

        # SynthID starts every hash chain at 1.
        hash_result = torch.ones(
            batch_size,
            dtype=torch.long,
            device=contexts.device,
        )

        # Hash the preceding context once per batch row; [B, context_width] -> [B]
        context_hash = self._accumulate_hash(
            hash_result,
            contexts,
        )

        # Extend each context hash with every possible next token, conceptually:
        #   candidate_hash[b, v] =
        #       hash(context[b] + [v])
        # [B] + [B, V] -> [B, V]
        candidate_hash = torch.vmap(
            self._accumulate_hash,
            in_dims=(None, 1),
            out_dims=1,
        )(
            context_hash,
            indices[:, :, None],
        )

        # Each SynthID key represents one watermarking "depth".
        # Reshape from [D] to [1, 1, D, 1] so PyTorch can broadcast it
        # against all batch rows and vocabulary entries.
        keys = self._get_keys(contexts.device)[None, None, :, None]

        # Append each watermark key to every candidate hash; [B, V] + [D] -> [B, V, D]
        return torch.vmap(
            self._accumulate_hash,
            in_dims=(None, 2),
            out_dims=2,
        )(
            candidate_hash,
            keys,
        )

    def _sample_g_values(
        self,
        ngram_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Map SynthID keys deterministically to binary g-values.

        ``ngram_keys`` has shape [B, V, D].

        Hugging Face SynthID pre-generates a binary sampling table. Each key
        indexes into that table modulo its size, producing a deterministic
        0 or 1 g-value.
        """
        sampling_table = self._get_sampling_table(ngram_keys.device)

        # [T] -> [1, 1, T] so one table can be used for all rows & depths.
        sampling_table = sampling_table.reshape(
            1,
            1,
            self.sampling_table_size,
        )

        # Convert arbitrary int64 hashes into valid sampling-table indices.
        indices = ngram_keys % self.sampling_table_size

        # Look up one binary g-value for every [batch, token, depth].
        return torch.take_along_dim(
            sampling_table,
            indices=indices,
            dim=2,
        )

    @staticmethod
    def _update_scores(
        scores: torch.Tensor,
        g_values: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SynthID's probability reweighting.

        Args:
            scores:
                Model logits, shape [B, V].

            g_values:
                Binary SynthID values, shape [B, V, D].

        Returns:
            Watermarked log-probabilities with shape [B, V].

        SynthID applies the reweighting once for each watermark depth.
        This implementation intentionally mirrors HuggingFace so that
        generated text remains behaviorally compatible with its SynthID
        implementation.

        """
        depth = g_values.shape[-1]

        # Convert model logits into probabilities over the vocabulary.
        probs = torch.softmax(scores, dim=1)

        for i in range(depth):
            # Select this watermark depth: [B, V, D] -> [B, V]
            g_values_at_depth = g_values[:, :, i]

            # Total probability mass currently assigned to tokens whose g-value is 1:
            # [B, V] * [B, V] -> sum over V -> [B, 1]
            g_mass = (g_values_at_depth * probs).sum(
                dim=1,
                keepdim=True,
            )

            # SynthID's probability-preserving reweighting step.
            probs = probs * (1 + g_values_at_depth - g_mass)

        # convert final probability distribution back into log-space.
        log_probs = torch.log(probs)

        # Replace log(0)=-inf with the smallest value of the tensor's dtype.
        return torch.where(
            torch.isfinite(log_probs),
            log_probs,
            torch.finfo(log_probs.dtype).min,
        )

    def watermark_logits(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SynthID to a batch of model logits."""
        batch_size, vocab_size = logits.shape

        # Candidate token IDs 0..V-1, repeated for every batch row. Shape: [B, V]
        indices = (
            torch.arange(
                vocab_size,
                device=logits.device,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        # Get deterministic SynthID keys per [request, candidate token, watermark depth]
        # and convert into watermark signals.
        g_values = self._sample_g_values(
            self._compute_keys(
                contexts,
                indices,
            )
        )

        return self._update_scores(
            logits,
            g_values,
        )

    def sample(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
        random_sampler: RandomSampler | None = None,
        skip_mask: torch.Tensor | None = None,
    ) -> WatermarkSample:
        if random_sampler is None:
            raise ValueError("SynthID requires a random sampler")

        watermarked_logits = self.watermark_logits(logits, contexts)

        if skip_mask is not None:
            watermarked_logits = torch.where(
                skip_mask.unsqueeze(-1),
                logits,
                watermarked_logits,
            )

        return WatermarkSample(
            token_ids=random_sampler(watermarked_logits),
            logits=watermarked_logits,
        )

    def _sample_watermarked(
        self,
        logits: torch.Tensor,
        contexts: torch.Tensor,
    ) -> WatermarkSample:
        raise ValueError("SynthID requires a random sampler")
