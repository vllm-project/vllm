class InputBatch:
    # ...

    def _make_sampling_metadata(self) -> SamplingMetadata:
        # ...

        if self.allowed_token_ids_mask_cpu_tensor is not None:
            # False means unrestricted: do not fill any logits with -inf.
            metadata_mask = self.allowed_token_ids_mask_cpu_tensor[req_index]
        else:
            metadata_mask = None

        # ...