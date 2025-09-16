from src.BPETokenizer import BPETokenizer


class TestBPETokenizer:
    def test_tokenizer_initialization(self):
        """Test that the tokenizer initializes with empty vocabularies and merges."""
        tokenizer = BPETokenizer()
        assert isinstance(tokenizer.vocab, dict), "Vocab should be a dictionary"
        assert isinstance(
            tokenizer.inverse_vocab, dict
        ), "Inverse vocab should be a dictionary"
        assert isinstance(
            tokenizer.bpe_merges, dict
        ), "BPE merges should be a dictionary"
        assert len(tokenizer.vocab) == 0, "Initial vocab should be empty"
        assert (
            len(tokenizer.inverse_vocab) == 0
        ), "Initial inverse vocab should be empty"
        assert len(tokenizer.bpe_merges) == 0, "Initial BPE merges should be empty"
