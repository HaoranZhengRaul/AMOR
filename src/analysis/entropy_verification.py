"""
CRITICAL: Verify entropy signal BEFORE building the gate.

If entropy doesn't correlate with retrieval need, the whole approach fails.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def verify_entropy_signal(ssm_model, dataset, device='cuda'):
    """
    CRITICAL CHECK: Does SSM entropy correlate with retrieval need?

    Returns:
        gap: entropy_at_retrieval - entropy_at_local
        Should be POSITIVE (> 0.1) for approach to work
    """
    ssm_model.eval()
    ssm_model = ssm_model.to(device)
    loader = DataLoader(dataset, batch_size=32)

    all_entropy_retrieval = []
    all_entropy_local = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Verifying entropy signal"):
            tokens = batch['tokens'].to(device)
            needs_retrieval = batch['needs_retrieval'].to(device)

            x = tokens[:, :-1]
            nr = needs_retrieval[:, 1:]

            logits, _ = ssm_model(x, return_hidden=True)

            # Compute entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(-1)

            # Split by retrieval need
            if nr.any():
                all_entropy_retrieval.extend(entropy[nr].cpu().numpy().tolist())
            if (~nr).any():
                all_entropy_local.extend(entropy[~nr].cpu().numpy().tolist())

    mean_retrieval = np.mean(all_entropy_retrieval) if all_entropy_retrieval else 0
    mean_local = np.mean(all_entropy_local) if all_entropy_local else 0
    gap = mean_retrieval - mean_local

    print("\n" + "="*50)
    print("ENTROPY VERIFICATION RESULTS")
    print("="*50)
    print(f"Entropy at RETRIEVAL positions: {mean_retrieval:.4f}")
    print(f"Entropy at LOCAL positions:     {mean_local:.4f}")
    print(f"GAP:                            {gap:.4f}")
    print("="*50)

    if gap > 0.1:
        print("GOOD: Entropy spikes at retrieval positions!")
        print("  -> Use gate_method='entropy' (primary approach)")
    elif gap > 0:
        print("MARGINAL: Small positive gap")
        print("  -> Try entropy, but have learned_ste as backup")
    else:
        print("BAD: Entropy does NOT correlate with retrieval")
        print("  -> Fall back to gate_method='learned_ste'")

    print("="*50 + "\n")

    return {
        'entropy_at_retrieval': float(mean_retrieval),
        'entropy_at_local': float(mean_local),
        'gap': float(gap),
        'use_entropy': bool(gap > 0.1),
        'n_retrieval_samples': len(all_entropy_retrieval),
        'n_local_samples': len(all_entropy_local),
    }
