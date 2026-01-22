"""
Simple Retrieval Task for initial debugging.
Use NeedleHaystackTask for paper experiments.
"""
import hashlib
import torch
import random
from torch.utils.data import Dataset


class RetrievalTaskDataset(Dataset):
    """
    Sequence structure:
    - Local patterns: Predictable from recent context (SSM can handle)
    - Retrieval patterns: Must copy from distant marked position

    Example:
    [A B A B] [M X] [A B A B] [R ?] [A B A B]
              ^              ^
           Marker       Retrieve X
           stores X
    """

    def __init__(
        self,
        size: int = 10000,
        seq_length: int = 128,
        vocab_size: int = 8,
        local_order: int = 2,
        retrieval_distance_range: tuple = (20, 50),
        retrieval_prob: float = 0.15,
        seed: int = None,
    ):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.local_order = local_order
        self.retrieval_distance_range = retrieval_distance_range
        self.retrieval_prob = retrieval_prob

        # Special tokens
        self.MARKER = vocab_size
        self.RETRIEVE = vocab_size + 1
        self.PAD = vocab_size + 2
        self.total_vocab = vocab_size + 3

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.data = [self._generate_sequence() for _ in range(size)]

    def _local_rule(self, context: list) -> int:
        """Deterministic local pattern: hash of context mod vocab_size.

        Uses hashlib.md5 for cross-session reproducibility.
        Python's built-in hash() is NOT deterministic across sessions.
        """
        context_bytes = str(context).encode('utf-8')
        h = int(hashlib.md5(context_bytes).hexdigest(), 16) % self.vocab_size
        return h

    def _generate_sequence(self):
        tokens = []
        needs_retrieval = []
        retrieval_source = []

        stored_value = None
        stored_position = None
        pending_retrieve = False

        t = 0
        while t < self.seq_length:
            if pending_retrieve and t < self.seq_length:
                tokens.append(stored_value)
                needs_retrieval.append(True)
                retrieval_source.append(stored_position)
                pending_retrieve = False
                stored_value = None
                t += 1
                continue

            if (stored_value is None and
                random.random() < self.retrieval_prob and
                t < self.seq_length - 20):

                tokens.append(self.MARKER)
                needs_retrieval.append(False)
                retrieval_source.append(-1)
                t += 1

                if t < self.seq_length:
                    stored_value = random.randint(0, self.vocab_size - 1)
                    stored_position = t
                    tokens.append(stored_value)
                    needs_retrieval.append(False)
                    retrieval_source.append(-1)
                    t += 1
                continue

            if (stored_value is not None and
                t - stored_position >= self.retrieval_distance_range[0] and
                random.random() < 0.3):

                tokens.append(self.RETRIEVE)
                needs_retrieval.append(False)
                retrieval_source.append(-1)
                pending_retrieve = True
                t += 1
                continue

            if len(tokens) >= self.local_order:
                context = []
                for i in range(self.local_order):
                    prev_tok = tokens[-(self.local_order - i)]
                    if prev_tok < self.vocab_size:
                        context.append(prev_tok)
                    else:
                        context.append(0)
                token = self._local_rule(context)
            else:
                token = random.randint(0, self.vocab_size - 1)

            tokens.append(token)
            needs_retrieval.append(False)
            retrieval_source.append(-1)
            t += 1

        return {
            'tokens': torch.tensor(tokens[:self.seq_length], dtype=torch.long),
            'needs_retrieval': torch.tensor(needs_retrieval[:self.seq_length], dtype=torch.bool),
            'retrieval_source': torch.tensor(retrieval_source[:self.seq_length], dtype=torch.long),
        }

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_stats(self):
        total_tokens = 0
        retrieval_tokens = 0
        for item in self.data:
            total_tokens += len(item['tokens'])
            retrieval_tokens += item['needs_retrieval'].sum().item()

        return {
            'total_tokens': total_tokens,
            'retrieval_tokens': retrieval_tokens,
            'retrieval_rate': retrieval_tokens / total_tokens,
        }
