from dataclasses import dataclass

import torch

from scripts.esmfold2.precompute_esmc_hidden_states import (
    _index_path_string,
    compute_hidden_states,
)


@dataclass
class FakeESMCOutput:
    hidden_states: torch.Tensor


class FakeESMC:
    def _tokenize(self, sequences):
        return torch.arange(len(sequences[0]) + 2).unsqueeze(0)

    def embed(self, tokens):
        return tokens.float().unsqueeze(-1).repeat(1, 1, 3)

    def __call__(self, *, sequence_tokens):
        layers = []
        for layer_idx in range(2):
            layers.append(
                sequence_tokens.float().unsqueeze(-1).repeat(1, 1, 3) + 100 * (layer_idx + 1)
            )
        return FakeESMCOutput(hidden_states=torch.stack(layers, dim=0))


def test_precompute_hidden_states_includes_initial_embedding_layer():
    hidden = compute_hidden_states(FakeESMC(), "ACD", torch.float32)
    assert list(hidden.shape) == [3, 3, 3]
    assert torch.equal(hidden[:, 0], torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))
    assert torch.equal(
        hidden[:, 1],
        torch.tensor([[101.0, 101.0, 101.0], [102.0, 102.0, 102.0], [103.0, 103.0, 103.0]]),
    )


def test_index_path_string_is_relative_to_cache_index(tmp_path):
    cache_index = tmp_path / "indices" / "cache_index.jsonl"
    hidden_path = tmp_path / "cache" / "nanobody.pt"
    expected = "../cache/nanobody.pt"
    assert _index_path_string(hidden_path, cache_index) == expected
