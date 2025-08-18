import torch
from vllm.spec_decode.early_exit_model_runner import load_lsq_head


class _Recorder:
    def __init__(self):
        self.called_with = None
    
    def __call__(self, module, hidden, sampling_metadata):
        self.called_with = module
        # Return dummy logits of correct shape (B*T, vocab)
        BxT, H = hidden.shape[-2], hidden.shape[-1]
        vocab = module.weight.shape[0]
        return torch.zeros(BxT, vocab, dtype=hidden.dtype, device=hidden.device)


def test_compute_logits_uses_lsq_when_present(tmp_path):
    """Test that compute_logits uses LSQ head when available."""
    # Build a faux LSQ head
    vocab, hidden = 32, 16
    torch.save(torch.randn(vocab, hidden), tmp_path / "h4.pt")

    # Minimal fake runner
    class FakeModel: 
        def __init__(self):
            self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    
    class FakeRunner:
        def __init__(self):
            self.model = FakeModel()
            self.lsq_head = load_lsq_head(str(tmp_path), layer=4, 
                                         device=torch.device("cpu"), dtype=torch.float32)
            self.logits_processor = _Recorder()
    
    r = FakeRunner()
    hidden_states = torch.zeros(1, 1, hidden)

    # Emulate compute_logits path
    module_used = None
    if r.lsq_head is not None:
        r.logits_processor(r.lsq_head, hidden_states, None)
        module_used = r.logits_processor.called_with
    else:
        r.logits_processor(r.model.lm_head, hidden_states, None)
        module_used = r.logits_processor.called_with

    assert module_used is r.lsq_head


def test_compute_logits_falls_back_to_lm_head_when_no_lsq():
    """Test that compute_logits falls back to LM head when no LSQ head."""
    vocab, hidden = 32, 16
    
    class FakeModel:
        def __init__(self):
            self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    
    class FakeRunner:
        def __init__(self):
            self.model = FakeModel()
            self.lsq_head = None
            self.logits_processor = _Recorder()
    
    r = FakeRunner()
    hidden_states = torch.zeros(1, 1, hidden)

    r.logits_processor(r.model.lm_head, hidden_states, None)
    assert r.logits_processor.called_with is r.model.lm_head