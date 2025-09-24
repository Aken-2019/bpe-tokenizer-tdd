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
        Encode text into a sequence of token IDs using byte-level BPE.
        
        Args:
            text (str): The text to encode.
            allowed_specials (set[str], optional): Set of special tokens that are allowed.
            
        Returns:
            list[int]: A sequence of token IDs representing the encoded text.
            
        Raises:
            ValueError: If the tokenizer hasn't been trained or special tokens not in vocab.
        """
        if len(self.vocab) == 0:
            raise ValueError("Tokenizer not trained.")
        if len(text) == 0:
            return []
        
        # Handle special tokens by splitting text
        if allowed_specials:
            # Validate all special tokens exist in vocabulary
            for special in allowed_specials:
                if special not in self.inverse_vocab:
                    raise ValueError(f"Special token {special} not found in vocabulary.")
            
            # Check if entire text is a special token
            if text in allowed_specials and text in self.inverse_vocab:
                return [self.inverse_vocab[text]]
            
            # Split text by special tokens and process each segment
            result = []
            remaining_text = text
            
            # Find special tokens in text and split
            segments = []
            current_pos = 0
            
            while current_pos < len(remaining_text):
                found_special = None
                found_pos = len(remaining_text)
                
                # Find the earliest occurring special token
                for special in allowed_specials:
                    pos = remaining_text.find(special, current_pos)
                    if pos != -1 and pos < found_pos:
                        found_pos = pos
                        found_special = special
                
                if found_special:
                    # Add text before special token (if any)
                    if found_pos > current_pos:
                        segments.append(remaining_text[current_pos:found_pos])
                    # Add the special token
                    segments.append(found_special)
                    current_pos = found_pos + len(found_special)
                else:
                    # No more special tokens, add remaining text
                    if current_pos < len(remaining_text):
                        segments.append(remaining_text[current_pos:])
                    break
            
            # Process each segment
            for segment in segments:
                if segment in allowed_specials:
                    # Special token
                    result.append(self.inverse_vocab[segment])
                else:
                    # Regular text
                    # Check for disallowed special tokens
                    for token in self.inverse_vocab:
                        if token.startswith("<|") and token.endswith("|>") and token in segment:
                            if token not in allowed_specials:
                                raise ValueError(f"Disallowed special token {token} found in text.")
                    
                    # Preprocess and encode regular text
                    processed_text = []
                    for i, char in enumerate(segment):
                        if char == " " and i != 0:
                            processed_text.append("Ġ")
                        if char != " ":
                            processed_text.append(char)
                    processed_text = "".join(processed_text)
                    
                    token_ids = [self.inverse_vocab[char] for char in processed_text]
                    result.extend(self.apply_merges(token_ids))
            
            return result
        else:
            # No allowed_specials - check for any special tokens and raise error
            for token in self.inverse_vocab:
                if token.startswith("<|") and token.endswith("|>") and token in text:
                    raise ValueError(f"Special token {token} found in text but not allowed.")

        # Preprocess: Replace spaces with "Ġ" (except at the beginning)
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Convert to token IDs using vocabulary and apply BPE merges
        token_ids = [self.inverse_vocab[char] for char in processed_text]
        return self.apply_merges(token_ids)
    
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
        if len(token_ids) == 0:
            return ""
            
        if not set(token_ids) <= set(self.vocab):
            raise ValueError(f"Unknown token id {set(token_ids) - set(self.vocab)}") 

        # Decode tokens to text
        text = "".join([self.vocab[id] for id in token_ids])
        
        # Convert "Ġ" back to spaces
        text = text.replace("Ġ", " ")
        
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
        # Preprocess: Replace spaces with "Ġ" (except at the beginning)
        processed_text = []
        for i, char in enumerate(text):
            if char == " " and i != 0:
                processed_text.append("Ġ")
            if char != " ":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        # Initialize vocab with all 256 byte values
        self.vocab = {i: chr(i) for i in range(256)}
        self.inverse_vocab = {chr(i): i for i in range(256)}
        
        # Add the "Ġ" character if it's not already in the vocab (it's needed for space preprocessing)
        if "Ġ" not in self.inverse_vocab:
            new_id = len(self.vocab)
            self.vocab[new_id] = "Ġ"
            self.inverse_vocab["Ġ"] = new_id

        # Add special tokens before training
        if allowed_special:
            for special in allowed_special:
                if special not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = special
                    self.inverse_vocab[special] = new_id

        # Convert processed text to token IDs using vocabulary
        token_ids = [self.inverse_vocab[char] for char in processed_text]

        # BPE merge process
        while len(self.vocab) < vocab_size:
            pair = self.find_freq_pair(token_ids)
            if pair is None:
                break
            else:
                pair_id = max(self.vocab) + 1
                self.bpe_merges[pair] = pair_id

                # Create merged token string
                merged_token = self.vocab[pair[0]] + self.vocab[pair[1]]
                self.vocab[pair_id] = merged_token
                self.inverse_vocab[merged_token] = pair_id
                token_ids = self.replace_pair(token_ids, pair, pair_id)

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
