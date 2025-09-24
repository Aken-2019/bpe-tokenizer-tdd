from collections import Counter, deque


class BPETokenizer:
    """A simple implementation of Byte Pair Encoding (BPE) tokenizer using byte-level encoding."""

    def __init__(self):
        """Initialize the BPE Tokenizer with empty vocabularies and merges."""
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token1, token2): merged_token_id}
        self.bpe_merges = {}
        
    def encode(self, text: str, allowed_specials: set[str] = None) -> list[int]:
        """
        Encode text into a sequence of token IDs using byte-level BPE.
        
        Args:
            text (str): The text to encode.
            allowed_specials (set[str], optional): Set of special tokens that are allowed.
            
        Returns:
            list[int]: A sequence of token IDs representing the encoded text.
            
        Raises:
            ValueError: If the tokenizer hasn't been trained or special tokens not in vocab.
        """
        pass
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text using byte-level BPE.
        
        Args:
            token_ids (list[int]): The sequence of token IDs to decode
            
        Returns:
            str: The decoded text with "Ġ" converted back to spaces
            
        Raises:
            ValueError: If any token ID is not in the vocabulary
        """
        pass
        
    def apply_merges(self, token_ids: list[int]) -> list[int]:
        """
        Apply learned BPE merges to a sequence of token IDs.
        
        Args:
            token_ids (list[int]): Initial sequence of token IDs
            
        Returns:
            list[int]: Sequence after applying all possible merges
        """
        pass

    def train(
        self, text: str, vocab_size: int, allowed_special: set[str] = None
    ) -> None:
        """
        Train the BPE tokenizer on the provided text using byte-level encoding.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set[str], optional): Special tokens to include in the vocabulary.

        Implementation note:
        This implementation uses byte-level BPE instead of character-level BPE:
        1. test_train_builds_initial_vocab - Building initial byte vocabulary (256 bytes)
        2. test_train_learns_merges_and_respects_vocab_size - BPE merge process
        3. test_train_adds_special_tokens - Special token handling
        """
        pass

    @staticmethod
    def find_freq_pair(token_ids: list[int]) -> tuple[int, int] | None:
        """
        Find the most frequent pair of adjacent tokens in the sequence.
        
        Args:
            token_ids (list[int]): Sequence of token IDs
            
        Returns:
            tuple[int, int] | None: Most frequent pair, or None if no pairs exist
        """
        if len(token_ids) < 2:
            return None
        
        pairs = Counter(zip(token_ids, token_ids[1:]))
        if not pairs:
            return None
            
        return max(pairs.items(), key=lambda x: x[1])[0]

    @staticmethod
    def replace_pair(
        token_ids: list[int], pair_to_replace: tuple[int, int], new_id: int
    ) -> list[int]:
        """
        Replace all occurrences of a token pair with a new token ID.
        
        Args:
            token_ids (list[int]): Original sequence
            pair_to_replace (tuple[int, int]): Pair of token IDs to replace
            new_id (int): New token ID to replace the pair with
            
        Returns:
            list[int]: Sequence with pairs replaced
        """
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_to_replace:
                replaced.append(new_id)
                # Remove the 2nd token of the pair, 1st was already removed
                dq.popleft()
            else:
                replaced.append(current)

        return replaced