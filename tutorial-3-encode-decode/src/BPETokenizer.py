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
        
    def encode(self, text: str, allowed_specials: set[str] = None) -> list[int]:
        """
        Encode text into a sequence of token IDs using the trained BPE merges.
        
        This method converts input text into tokens by:
        1. Handling special tokens if they are in allowed_specials
        2. Checking for unknown characters
        3. Applying BPE merges to convert characters into larger tokens
        
        Args:
            text (str): The text to encode. Can contain regular characters and special tokens.
            allowed_specials (set[str], optional): Set of special tokens that are allowed
                in this encoding operation. If None or if a special token is not in this set
                but appears in the text, all characters will be treated as regular text.
                For example, if "<|endoftext|>" is in allowed_specials and text, it will
                be encoded as a single token.
            
        Returns:
            list[int]: A sequence of token IDs representing the encoded text. For example,
                "hello" might be encoded as [23, 47, 61] after BPE merges are applied.
            
        Raises:
            ValueError: If the tokenizer hasn't been trained (vocab is empty)
            ValueError: If the text contains characters not present in the vocabulary
            ValueError: If any special token in allowed_specials is not in the vocabulary
            
        Example:
            >>> tokenizer = BPETokenizer()
            >>> tokenizer.train("hello world", vocab_size=10)
            >>> tokenizer.encode("hello")  # Returns something like [23, 47, 61]
            >>> tokenizer.encode("hi", allowed_specials={"<|endoftext|>"})  # Safely encode with special token awareness
        """
        if len(self.vocab) == 0:
            raise ValueError("Tokenizer not trained.")
        if len(text) == 0:
            return []
        
        # Step 1: Handle special tokens if provided in allowed_specials
        if allowed_specials:
            # Validate all special tokens exist in vocabulary
            if not set(allowed_specials) <= set(self.inverse_vocab):
                raise ValueError(f"Special tokens {set(allowed_specials) - set(self.inverse_vocab)} not found in vocabulary.")
            else:
                # Initialize processing queue with (text_segment, token_id) pairs
                # token_id is None for regular text, actual token_id for special tokens
                splitted_text_token_mapping = deque([(text, None)])
                special_encoded_split = []

                # Process each segment, looking for special tokens
                while len(splitted_text_token_mapping) > 0:
                    for special in allowed_specials:
                        left_text, pre_token = splitted_text_token_mapping.popleft()
                        print(left_text, pre_token)
                        
                        # Only look for special tokens in unprocessed text segments
                        if pre_token is None and special in left_text:
                            # Split text on special token boundaries
                            left_split = left_text.split(special)
                            print(left_split)

                            # Case 1: Text is exactly the special token
                            if left_split == ['']:
                                special_encoded_split.append((special, self.inverse_vocab[special]))
                            # Case 2: Text contains special token and regular text
                            else:
                                for i, itext in enumerate(left_split):
                                    # Handle regular text segments
                                    if itext != '':
                                        special_encoded_split.append((itext, None))
                                    
                                    # Insert special token between text segments
                                    # Don't add after the last segment
                                    if i < len(left_split) - 1:
                                        special_encoded_split.append((special, self.inverse_vocab[special]))
                        else:
                            # Pass through already processed segments unchanged
                            special_encoded_split.append((left_text, pre_token))
        else:
            # No special tokens - treat entire input as regular text
            special_encoded_split = [(text, None)]

        # Step 2: Validate all regular text characters are in vocabulary
        special_free_characters = "".join([text for text, pre_token in special_encoded_split if pre_token is None])
        print(special_free_characters)
        if not set(list(special_free_characters)) <= set(list(self.inverse_vocab)):
            print(set(list(special_free_characters)))
            raise ValueError("Unknown characters in text.")

        # Step 3: Encode all segments
        results = []
        for text, pre_token in special_encoded_split:
            if pre_token is not None:
                # Special tokens map directly to their vocabulary ID
                results.append(pre_token)
            else:
                # Regular text needs BPE encoding
                results.extend(self.apply_merges([self.inverse_vocab[t] for t in text]))

        return results
    
    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into text.
        
        Args:
            token_ids (list[int]): The sequence of token IDs to decode
            
        Returns:
            str: The decoded text
            
        Raises:
            ValueError: If any token ID is not in the vocabulary
        """
        if not set(token_ids) <= set(self.vocab):
            raise ValueError(f"Unknown token id {set(token_ids)- set(self.vocab)}") 

        text = "".join([self.vocab[id] for id in token_ids])
        return text
        
    def apply_merges(self, token_ids: list[int]) -> list[int]:
        """
        Apply learned BPE merges to a sequence of token IDs.
        
        Args:
            token_ids (list[int]): Initial sequence of token IDs
            
        Returns:
            list[int]: Sequence after applying all possible merges
            
        This is a helper method that implements the core BPE algorithm:
        1. Start with a sequence of token IDs
        2. Find any adjacent pairs that have a merge rule
        3. Apply the merge by replacing the pair with its merged token ID
        4. Repeat until no more merges are possible
        """
        encoded_ids = []
        token_ids = deque(token_ids)
        # Handle empty or single token case
        if len(token_ids) <= 1:
            return list(token_ids)

        # Process tokens by looking for merges
        while len(token_ids) >= 2:
            left_id = token_ids.popleft()
            if (left_id, token_ids[0]) in self.bpe_merges:
                encoded_ids.append(self.bpe_merges[(left_id, token_ids[0])])
                token_ids.popleft()
            else:
                encoded_ids.append(left_id)
        
        # Append any remaining token
        encoded_ids.extend(token_ids)
        return encoded_ids

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
        # build the initial vocab without BPE
        # pass `test_train_builds_initial_vocab`
        unique_char = sorted(list(set(text)))
        self.vocab = {i: ch for i, ch in enumerate(unique_char)}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}

        # pass `test_train_learns_merges_and_respects_vocab_size`
        # iterate vocab and build merges using BPE
        token_ids = [self.inverse_vocab[ch] for ch in text]
        while len(self.vocab) < vocab_size:
            pair = self.find_freq_pair(token_ids)
            print(pair)
            if pair is None:
                break
            else:
                pair_id = max(self.vocab) + 1
                self.bpe_merges[pair] = pair_id

                self.vocab[pair_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
                self.inverse_vocab[self.vocab[pair_id]] = pair_id
                token_ids = self.replace_pair(token_ids, pair, pair_id)

        # pass `test_train_adds_special_tokens`
        # handle special tokens
        if allowed_special:
            for special in allowed_special:
                new_id = max(self.vocab) + 1
                self.vocab[new_id] = special
                self.inverse_vocab[special] = new_id

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

        # Special cases to handle empty string or single char
        # Link to test `test_find_freq_pair_empty`
        if len(token_ids) < 2:
            return None

        # Count the pairs and return the most frequent one
        pair_counter = Counter(zip(token_ids, token_ids[1:]))
        max_pair = max(pair_counter, key=pair_counter.get)
        if pair_counter[max_pair] > 1:
            return max_pair
        else:
            return None

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
        new_token_ids = []
        token_ids_processing = deque(token_ids)

        # Iterate through the token_ids using a deque for efficient popping from the left
        while len(token_ids_processing) > 0:
            left_id = token_ids_processing.popleft()
            # Check if the next token forms the target pair with the current token
            if (
                len(token_ids_processing) > 0
                and (left_id, token_ids_processing[0]) == pair_to_replace
            ):
                # If so, append the new_id (merged token) and skip the next token
                new_token_ids.append(new_id)
                token_ids_processing.popleft()
            else:
                # Otherwise, just append the current token
                new_token_ids.append(left_id)

        return new_token_ids
