"""
Needle-in-a-Haystack Task: GUARANTEES SSM failure.

This task tests TRUE metacognition:
- Gate should fire when retrieval IS possible
- Gate should NOT fire when retrieval is IMPOSSIBLE
"""
import torch
import random
from torch.utils.data import Dataset


class NeedleHaystackTask(Dataset):
    """
    Associative recall with noise that guarantees SSM state decay.

    Setup:
    - Store key-value pairs: STORE key value
    - Add noise to decay SSM state
    - Query: QUERY key -> target is value

    The Metacognitive Twist:
    - Sometimes query is IMPOSSIBLE (key never stored)
    - Gate should NOT fire for impossible queries!
    """

    def __init__(
        self,
        size: int = 10000,
        seq_length: int = 256,
        vocab_size: int = 16,
        noise_length_range: tuple = (50, 150),
        impossible_query_prob: float = 0.2,
        seed: int = None,
    ):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.noise_length_range = noise_length_range
        self.impossible_query_prob = impossible_query_prob

        # Special tokens
        self.STORE = vocab_size
        self.QUERY = vocab_size + 1
        self.UNKNOWN = vocab_size + 2
        self.total_vocab = vocab_size + 3

        if seed:
            random.seed(seed)
            torch.manual_seed(seed)

        self.data = [self._generate() for _ in range(size)]

    def _generate(self):
        tokens = []
        needs_retrieval = []
        is_impossible = []

        stored_pairs = {}
        n_pairs = random.randint(2, 5)

        # Phase 1: Store key-value pairs
        for _ in range(n_pairs):
            key = random.randint(0, self.vocab_size - 1)
            value = random.randint(0, self.vocab_size - 1)
            stored_pairs[key] = value

            tokens.extend([self.STORE, key, value])
            needs_retrieval.extend([False, False, False])
            is_impossible.extend([False, False, False])

        # Phase 2: Noise (distracts SSM state)
        noise_len = random.randint(*self.noise_length_range)
        for _ in range(noise_len):
            tokens.append(random.randint(0, self.vocab_size - 1))
            needs_retrieval.append(False)
            is_impossible.append(False)

        # Phase 3: Queries
        n_queries = random.randint(3, 8)
        for _ in range(n_queries):
            if random.random() < self.impossible_query_prob:
                # Impossible query
                available_keys = set(range(self.vocab_size)) - set(stored_pairs.keys())
                if available_keys:
                    key = random.choice(list(available_keys))
                    tokens.extend([self.QUERY, key, self.UNKNOWN])
                    needs_retrieval.extend([False, False, False])  # Gate should NOT fire!
                    is_impossible.extend([False, False, True])
            else:
                # Possible query
                key = random.choice(list(stored_pairs.keys()))
                value = stored_pairs[key]
                tokens.extend([self.QUERY, key, value])
                needs_retrieval.extend([False, False, True])  # Gate SHOULD fire
                is_impossible.extend([False, False, False])

            # Noise between queries
            noise = random.randint(5, 20)
            for _ in range(noise):
                tokens.append(random.randint(0, self.vocab_size - 1))
                needs_retrieval.append(False)
                is_impossible.append(False)

        # Pad or truncate
        tokens = tokens[:self.seq_length]
        needs_retrieval = needs_retrieval[:self.seq_length]
        is_impossible = is_impossible[:self.seq_length]

        while len(tokens) < self.seq_length:
            tokens.append(random.randint(0, self.vocab_size - 1))
            needs_retrieval.append(False)
            is_impossible.append(False)

        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'needs_retrieval': torch.tensor(needs_retrieval, dtype=torch.bool),
            'is_impossible': torch.tensor(is_impossible, dtype=torch.bool),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_stats(self):
        """Return statistics about the dataset."""
        total_tokens = 0
        retrieval_tokens = 0
        impossible_tokens = 0
        for item in self.data:
            total_tokens += len(item['tokens'])
            retrieval_tokens += item['needs_retrieval'].sum().item()
            impossible_tokens += item['is_impossible'].sum().item()

        return {
            'total_tokens': total_tokens,
            'retrieval_tokens': retrieval_tokens,
            'retrieval_rate': retrieval_tokens / total_tokens,
            'impossible_tokens': impossible_tokens,
            'impossible_rate': impossible_tokens / total_tokens,
        }
