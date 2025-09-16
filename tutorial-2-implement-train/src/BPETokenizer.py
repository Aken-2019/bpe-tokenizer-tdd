from collections import Counter, deque

class BPETokenizer:
    """A simple implementation of Byte Pair Encoding (BPE) tokenizer."""

    def __init__(self):
        """Initialize the BPE Tokenizer with empty vocabularies and merges."""
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token1, token2): merged_token_id}
        self.bpe_merges = {}

    def train(
        self, text: str, vocab_size: int, allowed_special: set[str] = None
    ) -> None:
        """
        Train the BPE tokenizer on the provided text.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set[str], optional): Special tokens to include in the vocabulary.

        Implementation note:
        This implementation satisfies three test cases in the tutorial:
        1. test_train_builds_initial_vocab - Building initial character vocabulary
        2. test_train_learns_merges_and_respects_vocab_size - BPE merge process
        3. test_train_adds_special_tokens - Special token handling

        See the test file and tutorial for detailed expectations of each component.
        """
        pass

    @staticmethod
    def find_freq_pair(token_ids: list[int]) -> tuple[int, int] | None:
        """
        Find the most frequent adjacent pair of token IDs in the sequence.

        Args:
            token_ids (list[int]): A list of token IDs to search for pairs.

        Returns:
            tuple[int, int] | None: The most frequent pair of adjacent token IDs,
                or None if no pairs exist.

        Example:
            If token_ids is [1, 2, 1, 2, 3, 4], and 1,2 appears most frequently,
            this will return (1, 2).
        """
        pass

    @staticmethod
    def replace_pair(
        token_ids: list[int], pair_to_replace: tuple[int, int], new_id: int
    ) -> list[int]:
        """
        Replace all occurrences of a token pair with a new token ID.

        Args:
            token_ids (list[int]): The list of token IDs to process.
            pair_to_replace (tuple[int, int]): The pair of token IDs to replace.
            new_id (int): The new token ID to use as replacement.

        Returns:
            list[int]: A new list with all occurrences of the pair replaced.

        Example:
            token_ids = [1, 2, 3, 1, 2]
            pair_to_replace = (1, 2)
            new_id = 4
            Returns: [4, 3, 4]  # Both occurrences of (1,2) became 4
        """
        pass
