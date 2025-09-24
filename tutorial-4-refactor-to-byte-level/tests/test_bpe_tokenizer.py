import pytest
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
        """Test that the initial vocabulary is built correctly with byte-level encoding."""
        tokenizer = BPETokenizer()
        text = "hello"
        tokenizer.train(text, vocab_size=300)  # vocab_size larger than 256 byte values

        # In byte-level BPE, initial vocab should contain all 256 byte values
        for i in range(256):
            assert i in tokenizer.vocab, f"Byte value {i} should be in vocab"
            assert chr(i) == tokenizer.vocab[i], f"Byte {i} should map to chr({i})"
        
        # Check that text characters are properly mapped
        for char in text:
            assert char in tokenizer.inverse_vocab, f"Character '{char}' should be in inverse vocab"

    def test_train_learns_merges_and_respects_vocab_size(self):
        """Test that the tokenizer learns BPE merges correctly and respects vocab_size."""
        tokenizer = BPETokenizer()
        text = "ababab"  # Most frequent pair is ('a', 'b')

        # Byte-level BPE starts with 256 byte values + Ġ character
        initial_vocab_size = 257  # 256 bytes + Ġ
        vocab_size = initial_vocab_size + 1  # Allows for exactly one merge

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
        a_id = tokenizer.inverse_vocab["a"]  # Should be ord('a') = 97
        b_id = tokenizer.inverse_vocab["b"]  # Should be ord('b') = 98
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

        # Byte-level BPE starts with 256 byte values + Ġ character
        initial_vocab_size = 257  # 256 bytes + Ġ
        vocab_size = initial_vocab_size + 5  # Plenty of room, but no pairs to merge

        tokenizer.train(text, vocab_size)

        assert len(tokenizer.vocab) == initial_vocab_size, "Vocab size should be the initial 257 tokens (256 bytes + Ġ)"
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
        vocab_size = 260  # 256 bytes + 4 merges

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
        tokenizer.train("", 300)
        assert len(tokenizer.vocab) == 257, "Empty sequence should result in base byte vocab + Ġ"

        # Test training with special tokens
        tokenizer = BPETokenizer()
        special_tokens = {"<|endoftext|>", "<|pad|>"}
        tokenizer.train(text, vocab_size, allowed_special=special_tokens)

        for token in special_tokens:
            assert (
                token in tokenizer.inverse_vocab
            ), f"Special token {token} should be in vocab"

    def test_encode_basic(self):
        """Test encoding a simple string with learned tokens."""
        tokenizer = BPETokenizer()
        text = "hello"
        tokenizer.train(text, vocab_size=300)  # Large enough to avoid merges
        
        # Encoding the same text we trained on
        token_ids = tokenizer.encode(text)
        
        # Check that each character maps to its byte value
        expected_ids = [ord(c) for c in text]
        assert token_ids == expected_ids

    def test_encode_with_merges(self):
        """Test encoding when BPE merges are involved."""
        tokenizer = BPETokenizer()
        text = "ababab"
        tokenizer.train(text, vocab_size=258)  # 257 base + 1 merge for 'ab'
        
        # The string "ababab" should now encode to fewer tokens
        # since 'ab' will be treated as a single token
        token_ids = tokenizer.encode(text)
        assert len(token_ids) < len(text), "Encoding should be shorter than original text"

    def test_encode_empty_string(self):
        """Test encoding an empty string."""
        tokenizer = BPETokenizer()
        tokenizer.train("hello", vocab_size=300)
        assert tokenizer.encode("") == [], "Empty string should encode to empty list"

    def test_encode_unknown_chars(self):
        """Test encoding text with characters not seen during training - should work in byte-level BPE."""
        tokenizer = BPETokenizer()
        tokenizer.train("hello", vocab_size=300)
        
        # In byte-level BPE, all characters should be encodable since we start with 256 byte values
        token_ids = tokenizer.encode("world")  # Should not raise error
        expected_ids = [ord(c) for c in "world"]
        assert token_ids == expected_ids, "All characters should be encodable in byte-level BPE"

    def test_space_preprocessing(self):
        """Test that spaces are properly handled with Ġ preprocessing in byte-level BPE."""
        tokenizer = BPETokenizer()
        text = "hello world"
        tokenizer.train(text, vocab_size=300)
        
        token_ids = tokenizer.encode(text)
        
        # Space should be replaced with Ġ during preprocessing except at start
        # We expect: h, e, l, l, o, Ġ, w, o, r, l, d
        expected_ids = [ord(c) for c in "hello"] + [tokenizer.inverse_vocab["Ġ"]] + [ord(c) for c in "world"]
        assert token_ids == expected_ids, "Spaces should be converted to Ġ except at beginning"

    def test_decode_empty_sequence(self):
        """Test decoding an empty sequence."""
        tokenizer = BPETokenizer()
        tokenizer.train("hello", vocab_size=300)
        assert tokenizer.decode([]) == "", "Empty sequence should decode to empty string"

    def test_decode_unknown_tokens(self):
        """Test decoding with token IDs not in vocabulary."""
        tokenizer = BPETokenizer()
        tokenizer.train("hello", vocab_size=300)
        
        with pytest.raises(ValueError):
            tokenizer.decode([999])  # Token ID 999 doesn't exist

    def test_encode_with_special_tokens(self):
        """Test encoding with special tokens in the vocabulary."""
        tokenizer = BPETokenizer()
        special_token = "<|endoftext|>"
        tokenizer.train("hello", vocab_size=300, allowed_special={special_token})
        
        # Special token should encode to a single token ID when allowed
        token_ids = tokenizer.encode(special_token, allowed_specials={special_token})
        assert len(token_ids) == 1, "Special token should encode to single ID"
        assert token_ids[0] == tokenizer.inverse_vocab[special_token]
        
        # Special token should be treated as regular text when not allowed
        with pytest.raises(ValueError):
            tokenizer.encode(special_token)

    def test_encode_with_mixed_content(self):
        """Test encoding text that contains both regular text and special tokens."""
        tokenizer = BPETokenizer()
        special_token = "<|endoftext|>"
        tokenizer.train("hello", vocab_size=300, allowed_special={special_token})
        
        # Encode text with a special token
        text = f"hello{special_token}"
        token_ids = tokenizer.encode(text, allowed_specials={special_token})
        
        # Should encode 'hello' as normal and special token as single token
        expected_ids = [ord(c) for c in "hello"]  # Use ord() for byte-level
        expected_ids.append(tokenizer.inverse_vocab[special_token])
        assert token_ids == expected_ids

    def test_decode_with_special_tokens(self):
        """Test decoding sequences containing special tokens."""
        tokenizer = BPETokenizer()
        special_token = "<|endoftext|>"
        tokenizer.train("hello", vocab_size=300, allowed_special={special_token})
        
        # Get the token ID for the special token
        special_id = tokenizer.inverse_vocab[special_token]
        
        # Decode a sequence with the special token
        text = tokenizer.decode([special_id])
        assert text == special_token, "Special token should decode correctly"

    def test_apply_merges_basic(self):
        """Test basic merge rule application."""
        tokenizer = BPETokenizer()
        text = "ababab"
        tokenizer.train(text, vocab_size=258)  # 257 base + 1 merge for 'ab'
        
        # Initial token sequence before merges (using vocab token IDs)
        char_ids = [tokenizer.inverse_vocab[c] for c in text]
        
        # Apply merges
        merged_ids = tokenizer.apply_merges(char_ids)
        
        # Should use learned 'ab' merge to reduce sequence length
        assert len(merged_ids) < len(char_ids), "Merges should reduce sequence length"
        
        # Each pair of 'a','b' should be replaced with 'ab' token
        ab_token_id = tokenizer.inverse_vocab["ab"]
        expected_ids = [ab_token_id] * 3  # "ababab" -> ["ab", "ab", "ab"]
        assert merged_ids == expected_ids, "Should merge all 'ab' pairs"

    def test_apply_merges_no_applicable_rules(self):
        """Test when no merge rules can be applied."""
        tokenizer = BPETokenizer()
        text = "abc"
        tokenizer.train(text, vocab_size=257)  # No merges possible - just base vocab
        
        # Convert to token IDs (using vocab)
        token_ids = [tokenizer.inverse_vocab[c] for c in text]
        
        # Apply merges - should return same sequence since no merges are possible
        result = tokenizer.apply_merges(token_ids)
        assert result == token_ids, "Should return original sequence when no merges possible"

    def test_apply_merges_multiple_rules(self):
        """Test applying multiple merge rules in correct order."""
        tokenizer = BPETokenizer()
        text = "hello hello"  # Common pairs: 'll', 'he', etc.
        tokenizer.train(text, vocab_size=261)  # 257 base + 4 merges
        
        # Initial character sequence (using vocab token IDs)
        char_ids = [tokenizer.inverse_vocab[c] for c in "hello"]
        
        # Apply merges
        merged_ids = tokenizer.apply_merges(char_ids)
        
        # Should use all applicable merge rules
        assert len(merged_ids) < len(char_ids), "Should apply multiple merge rules"
        
        # Verify the result can be decoded back correctly
        decoded = tokenizer.decode(merged_ids)
        assert decoded == "hello", "Multiple merges should preserve original text"

    def test_apply_merges_empty_sequence(self):
        """Test applying merges to empty sequence."""
        tokenizer = BPETokenizer()
        tokenizer.train("hello", vocab_size=300)
        
        result = tokenizer.apply_merges([])
        assert result == [], "Should handle empty sequence"
