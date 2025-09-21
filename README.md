This folder contains a test-driven development (TDD) approach to building a Byte Pair Encoding (BPE) tokenizer. It provides a step-by-step tutorial to guide you through the process of implementing the tokenizer from Sebastian Raschka's blog post: [BPE Tokenizer from Scratch by Sebastian Raschka](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html). I recommend reading the post first to understand the underlying concepts.

Note: While this implementation follows the concepts from Sebastian Raschka's blog post, it has been reimplemented from scratch with a focus on test-driven development and educational purposes. The implementation differs from the original to better suit the tutorial format:
1. The code is organized into multiple tutorial sections, each focusing on a specific aspect of the BPE tokenizer.
2. Each section includes its own documentation, tests, and implementation templates to facilitate learning and practice. Docstring, type hints, and comments are added for clarity.
3. The original implementation is a byte-level BPE tokenizer, while this tutorial implements a character-level BPE tokenizer for simplicity.

## Requirements

- The code is developed with tested from Python 3.10 to 3.13.
- Dependencies listed in `requirements.txt` of each tutorial section
- You have a basic understanding of Python and test-driven development (TDD)

## Tutorial Structure

The project is divided into multiple tutorial sections, each building upon the previous one:

1. `tutorial-1-initialization/`: Setting up the project and implementing the basic tokenizer class
2. `tutorial-2-implement-train/`: Implementing the core BPE training algorithm
3. `tutorial-3-encode-decode/`: Implementing text encoding and decoding with the trained BPE model


Each tutorial section is self-contained with its own:
- Documentation in the `docs/` folder
- Test suite in `tests/`
- Implementation template in `src/BPETokenizer.py`
- Complete solution in `src/solution_BPETokenizer.py`

To follow the tutorial:
1. Start with `tutorial-1-initialization`
2. Read the documentation in `docs/`
3. Run the tests and implement the required functionality
4. Compare your solution with the provided one if needed
5. Move on to the next tutorial section

## Project Structure

The project is organized as follows:

```
tutorial-X-XXX/
    ├── docs/
    │   └── tutorial-X-XXX.md
    │   └── ...
    ├── src/
    │   └── BPETokenizer.py (template)
    │   └── solution_BPETokenizer.py
    └── tests/
        └── test_bpe_tokenizer.py
```

Every project folder contains the following components at different phases of the tutorial:

- `docs/`: Contains documentation files, including the tutorial.
- `src/BPETokenizer.py` (template) and `src/solution_BPETokenizer.py`: The main package containing the BPE tokenizer implementation.
- `tests/`: The directory containing unit tests for the tokenizer.

## Attribution

This tutorial and implementation are based on Sebastian Raschka's blog post [BPE Tokenizer from Scratch](https://sebastianraschka.com/blog/2025/bpe-from-scratch.html). The original implementation code is available in the `reference/` directory.



## Additional Resources

For a deeper understanding of BPE tokenization, check out these resources:

1. [Building a BPE Tokenizer From Scratch - Video Tutorial](https://www.youtube.com/watch?v=zduSFxRajkE) - A comprehensive video walkthrough of BPE implementation
2. [Stanford CS336: Natural Language Processing](https://stanford-cs336.github.io/spring2025/) - Covers tokenization and subword models in depth

These materials complement Sebastian Raschka's blog post and provide supplemental perspectives on BPE implementation.