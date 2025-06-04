import torch

from hanstream.model import CTCConfig, TinyCTC, train_step

torch.manual_seed(11)
torch.set_num_threads(1)
model = TinyCTC(CTCConfig(4, 16, 4, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
features = torch.randn(2, 12, 4)
lengths = torch.tensor([12, 10])
targets = torch.tensor([1, 2, 2, 1])
target_lengths = torch.tensor([2, 2])
losses = [
    train_step(model, optimizer, features, lengths, targets, target_lengths) for _ in range(40)
]
print({'initial_loss': losses[0], 'final_loss': losses[-1]})
assert losses[-1] < losses[0]
