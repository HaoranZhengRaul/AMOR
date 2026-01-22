"""
Training loops for AMOR v2.

PRIMARY: MoE-style with STE + balance loss
BACKUP: RL with REINFORCE

Training functions:
- train_ssm_baseline: Train standalone SSM model
- train_attention_baseline: Train full attention transformer
- train_amor_v2: Train AMOR v2 with entropy gate (alias for train_moe_style)
- evaluate: Evaluate model on dataset
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path


def train_ssm_baseline(model, dataset, epochs=10, batch_size=32, lr=1e-3, device='cuda'):
    """Train SSM baseline (needed for entropy verification)"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in tqdm(loader, desc=f'SSM Epoch {epoch+1}'):
            tokens = batch['tokens'].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            logits = model(x) if not hasattr(model, 'ssm') else model(x)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # Track accuracy
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_tokens
        print(f"  SSM Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

    return model


def train_moe_style(
    model,
    train_dataset,
    val_dataset,
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    balance_loss_weight: float = 0.1,
    device: str = 'cuda',
):
    """PRIMARY: MoE-style training with STE + balance loss."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    history = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_metrics = {}

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            tokens = batch['tokens'].to(device)
            needs_retrieval = batch['needs_retrieval'].to(device)
            is_impossible = batch.get('is_impossible', torch.zeros_like(needs_retrieval)).to(device)

            x = tokens[:, :-1]
            y = tokens[:, 1:]
            nr = needs_retrieval[:, 1:]
            imp = is_impossible[:, 1:]

            # For oracle gate, pass ground truth; for entropy gate, don't
            if hasattr(model, 'gate_method') and model.gate_method == 'oracle':
                outputs = model(x, ground_truth_gate=nr.float())
            else:
                outputs = model(x)

            task_loss = F.cross_entropy(
                outputs['logits'].reshape(-1, model.vocab_size),
                y.reshape(-1),
            )

            balance_loss = model.gate.balance_loss(outputs['gate_prob'])

            loss = task_loss + balance_loss_weight * balance_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                metrics = model.compute_metrics(outputs, y, nr, imp)
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v

                if outputs['entropy'] is not None:
                    entropy_stats = model.gate.entropy_analysis(outputs['entropy'], nr)
                    for k, v in entropy_stats.items():
                        if isinstance(v, torch.Tensor):
                            v = v.item()
                        epoch_metrics[k] = epoch_metrics.get(k, 0) + v

        n_batches = len(train_loader)
        epoch_loss /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        val_metrics = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}, "
              f"acc={epoch_metrics.get('accuracy', 0):.3f}, "
              f"gate_f1={epoch_metrics.get('gate_f1', 0):.3f}, "
              f"gate_rate={epoch_metrics.get('gate_fires', 0):.3f}, "
              f"val_acc={val_metrics.get('accuracy', 0):.3f}")

        if 'entropy_gap' in epoch_metrics:
            print(f"  Entropy gap: {epoch_metrics['entropy_gap']:.4f}")
        if 'metacognitive_gap' in epoch_metrics:
            print(f"  Metacognitive gap: {epoch_metrics['metacognitive_gap']:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            **{f'train_{k}': v for k, v in epoch_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

    return history


def evaluate(model, dataloader, device):
    model.eval()
    all_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            needs_retrieval = batch['needs_retrieval'].to(device)
            is_impossible = batch.get('is_impossible', torch.zeros_like(needs_retrieval)).to(device)

            x = tokens[:, :-1]
            y = tokens[:, 1:]
            nr = needs_retrieval[:, 1:]
            imp = is_impossible[:, 1:]

            # For oracle gate, pass ground truth; for entropy gate, don't
            if hasattr(model, 'gate_method') and model.gate_method == 'oracle':
                outputs = model(x, ground_truth_gate=nr.float())
            else:
                outputs = model(x)
            metrics = model.compute_metrics(outputs, y, nr, imp)

            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                all_metrics[k] = all_metrics.get(k, 0) + v

    n_batches = len(dataloader)
    for k in all_metrics:
        all_metrics[k] /= n_batches

    return all_metrics


def train_attention_baseline(model, dataset, epochs=10, batch_size=32, lr=1e-3, device='cpu'):
    """Train full attention baseline model."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in tqdm(loader, desc=f'Attn Epoch {epoch+1}'):
            tokens = batch['tokens'].to(device)
            x, y = tokens[:, :-1], tokens[:, 1:]

            logits = model(x)

            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / total_tokens
        print(f"  Attn Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        history.append({'epoch': epoch+1, 'loss': avg_loss, 'accuracy': accuracy})

    return model, history


# Alias for clarity
train_amor_v2 = train_moe_style


def save_results(results: dict, path: str):
    """Save results to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def load_results(path: str) -> dict:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
