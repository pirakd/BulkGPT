import tiktoken
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
from model import Transformer


def get_batch(data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.from_numpy(np.stack([data[i:i+block_size] for i in ix]))
    y = torch.from_numpy(np.stack([data[i+1:i+block_size+1] for i in ix]))
    return x, y


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    n_steps = 100000
    batch_size = 16
    block_size = 32
    learning_rate = 3e-4
    n_embd = 64
    n_att_heads = 4
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    shakespeare_text = read_data("data/shakespere/input.txt")
    enc = tiktoken.get_encoding(encoding_name="gpt2")
    token_ids = np.array(enc.encode_ordinary(shakespeare_text))

    n_vocab = np.max(token_ids) + 1
    model = Transformer(n_vocab, n_embd, block_size,
                        n_att_heads, device).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(n_steps):
        x, y = get_batch(token_ids, block_size, batch_size)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), y.view(B*T))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
        if (step % 1000 == 0 and step > 0) or step == n_steps - 1:
            # generate from the model
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(context, 100, block_size)
            text = enc.decode(generated.tolist()[0])
            print(text)
            # print(model.generate(context, 100, block_size).tolist())


if __name__ == "__main__":
    main()
