from vllm.config.speculative import SpeculativeConfig


def test_weight_filenames_are_not_custom_proposer_paths():
    assert not SpeculativeConfig._is_custom_proposer_path("draft.pt")
    assert not SpeculativeConfig._is_custom_proposer_path("model.safetensors")
    assert not SpeculativeConfig._is_custom_proposer_path("eagle.weights")


def test_python_import_paths_still_detected():
    assert SpeculativeConfig._is_custom_proposer_path("my_module.MyProposer")
    assert SpeculativeConfig._is_custom_proposer_path("pkg.sub.Proposer")
