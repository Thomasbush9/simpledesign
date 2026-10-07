import torch
from transformers import EsmTokenizer

from simpledesign.data.dataset import ProteinCollator


def test_collated_batch_supports_stride_preserving_transfer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("<cls>\n<pad>\n<eos>\n<unk>\nA\nG\n<mask>\n")
    tokenizer = EsmTokenizer(vocab_file=str(vocab))
    items = [
        {
            "id": str(i),
            "sequence": sequence,
            "coords": torch.zeros(len(sequence), 3),
            "b_factor": torch.full((len(sequence),), 90.0),
        }
        for i, sequence in enumerate(["AG", "AGA", "A", "GAG"])
    ]
    batch = ProteinCollator(tokenizer)(items)
    torch.testing.assert_close(batch["idx"], torch.arange(5).repeat(4, 1))
    for tensor in batch.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        # Pinned-memory transfer preserves strides; overlapping expanded rows must fail here.
        copied = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype)
        copied.copy_(tensor)
        torch.testing.assert_close(copied, tensor)
        if torch.cuda.is_available():
            pinned = tensor.pin_memory()
            assert pinned.is_pinned()
            torch.testing.assert_close(pinned, tensor)
