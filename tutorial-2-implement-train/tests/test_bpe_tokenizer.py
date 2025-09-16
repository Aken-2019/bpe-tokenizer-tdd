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

    def test_train_builds_initial_vocab(self):
        """Test that the initial vocabulary is built correctly from unique characters."""
        tokenizer = BPETokenizer()
        text = "hello"
        tokenizer.train(text, vocab_size=10)  # vocab_size is larger than unique chars

        unique_chars = sorted(list(set(text)))
        expected_vocab = {i: ch for i, ch in enumerate(unique_chars)}
        expected_inverse_vocab = {ch: i for i, ch in enumerate(unique_chars)}

        # Check that the initial character vocabulary is correct
        for i, char in expected_vocab.items():
            assert tokenizer.vocab[i] == char
        for char, i in expected_inverse_vocab.items():
            assert tokenizer.inverse_vocab[char] == i

    def test_train_learns_merges_and_respects_vocab_size(self):
        """Test that the tokenizer learns BPE merges correctly and respects vocab_size."""
        tokenizer = BPETokenizer()
        text = "ababab"  # Most frequent pair is ('a', 'b')

        initial_chars = sorted(list(set(text)))
        vocab_size = len(initial_chars) + 1  # Allows for exactly one merge

        tokenizer.train(text, vocab_size)

        assert (
            len(tokenizer.vocab) == vocab_size
        ), f"Vocabulary size should be {vocab_size}"
        assert len(tokenizer.bpe_merges) == 1, "Should have recorded exactly one merge"

        # Check that the new token 'ab' has been added
        assert (
            "ab" in tokenizer.inverse_vocab
        ), "Merged token 'ab' should be in the vocabulary"

        # Check that the merge was recorded correctly
        a_id = tokenizer.inverse_vocab["a"]
        b_id = tokenizer.inverse_vocab["b"]
        ab_id = tokenizer.inverse_vocab["ab"]

        assert (
            a_id,
            b_id,
        ) in tokenizer.bpe_merges, "The pair ('a', 'b') should be in merges"
        assert (
            tokenizer.bpe_merges[(a_id, b_id)] == ab_id
        ), "The merge should map to the correct new token ID"

    def test_train_no_merges_if_no_pairs(self):
        """Test that no merges are performed if no pairs exist."""
        tokenizer = BPETokenizer()
        text = "abcde"

        initial_chars = sorted(list(set(text)))
        vocab_size = len(initial_chars) + 5  # Plenty of room, but no pairs to merge

        tokenizer.train(text, vocab_size)

        assert len(tokenizer.vocab) == len(
            initial_chars
        ), "Vocab size should be the number of unique characters"
        assert len(tokenizer.bpe_merges) == 0, "No merges should be recorded"

    def test_find_freq_pair_basic(self):
        """Test finding most frequent pair in a simple sequence."""
        tokenizer = BPETokenizer()
        token_ids = [1, 2, 3, 1, 2]  # (1,2) appears twice
        most_freq = tokenizer.find_freq_pair(token_ids)
        assert most_freq == (1, 2), "Should find (1,2) as most frequent pair"

    def test_find_freq_pair_empty(self):
        """Test finding pairs in empty or single-token sequences."""
        tokenizer = BPETokenizer()
        assert tokenizer.find_freq_pair([]) is None, "Empty sequence should return None"
        assert tokenizer.find_freq_pair([1]) is None, "Single token should return None"

    def test_find_freq_pair_ties(self):
        """Test that when multiple pairs have same frequency, consistently picks one."""
        tokenizer = BPETokenizer()
        token_ids = [1, 2, 3, 4, 3, 4]  # (3,4) appears twice
        most_freq = tokenizer.find_freq_pair(token_ids)
        assert most_freq == (3, 4), "Should find (3,4) as most frequent pair"

    def test_replace_pair_basic(self):
        """Test basic pair replacement."""
        tokenizer = BPETokenizer()
        token_ids = [1, 2, 3, 1, 2]
        pair = (1, 2)
        new_id = 4
        result = tokenizer.replace_pair(token_ids, pair, new_id)
        assert result == [4, 3, 4], "Should replace both occurrences of (1,2) with 4"

    def test_replace_pair_no_matches(self):
        """Test replacement when pair isn't found."""
        tokenizer = BPETokenizer()
        token_ids = [1, 3, 2, 4]
        pair = (1, 2)
        new_id = 5
        result = tokenizer.replace_pair(token_ids, pair, new_id)
        assert result == token_ids, "Should not modify sequence when pair isn't found"

    def test_replace_pair_adjacent_pairs(self):
        """Test replacement of adjacent pairs."""
        tokenizer = BPETokenizer()
        token_ids = [1, 2, 1, 2]  # Two adjacent (1,2) pairs
        pair = (1, 2)
        new_id = 3
        result = tokenizer.replace_pair(token_ids, pair, new_id)
        assert result == [3, 3], "Should handle adjacent pairs correctly"

    def test_train_adds_special_tokens(self):
        """Test that special tokens are added to the vocabulary correctly."""
        tokenizer = BPETokenizer()
        text = "hi"
        special_token = "<|endoftext|>"

        # vocab_size is large enough to not interfere
        tokenizer.train(text, vocab_size=10, allowed_special={special_token})

        assert (
            special_token in tokenizer.inverse_vocab
        ), "Special token should be in the inverse vocabulary"

        # Check that the special token ID is unique and correctly mapped
        special_token_id = tokenizer.inverse_vocab[special_token]
        assert tokenizer.vocab[special_token_id] == special_token

        # Ensure it didn't displace a character
        assert "h" in tokenizer.inverse_vocab
        assert "i" in tokenizer.inverse_vocab

    def test_train_full_functionality(self):
        """Test the full training process with a single concatenated sequence."""
        tokenizer = BPETokenizer()
        text = "hello worldhello world"
        vocab_size = 15

        # Train the tokenizer
        tokenizer.train(text, vocab_size)

        # Verify vocabulary size
        assert len(tokenizer.vocab) <= vocab_size, "Vocab should not exceed target size"

        # Check that common character pairs were merged
        common_pairs = [("h", "e"), ("l", "l"), ("r", "l")]
        merged = False
        for first, second in common_pairs:
            if first + second in tokenizer.inverse_vocab:
                merged = True
                break
        assert merged, "Should have merged at least one common pair"

        # Test handling of empty sequence
        tokenizer = BPETokenizer()
        tokenizer.train("", 10)
        assert len(tokenizer.vocab) == 0, "Empty sequence should result in empty vocab"

        # Test training with special tokens
        tokenizer = BPETokenizer()
        special_tokens = {"<|endoftext|>", "<|pad|>"}
        tokenizer.train(text, vocab_size, allowed_special=special_tokens)

        for token in special_tokens:
            assert (
                token in tokenizer.inverse_vocab
            ), f"Special token {token} should be in vocab"
