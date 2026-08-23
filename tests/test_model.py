import pytest


@pytest.mark.model
def test_training_decreases_loss():
    import torch

    from hanstream.model import CTCConfig, TinyCTC, ctc_loss, train_step

    torch.manual_seed(11)
    torch.set_num_threads(1)
    model = TinyCTC(CTCConfig(4, 16, 4, 1))
    x = torch.randn(2, 12, 4)
    lengths = torch.tensor([12, 10])
    y = torch.tensor([1, 2, 2, 1])
    y_lengths = torch.tensor([2, 2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = float(ctc_loss(model(x, lengths), y, lengths, y_lengths).detach())
    for _ in range(35):
        train_step(model, optimizer, x, lengths, y, y_lengths)
    final = float(ctc_loss(model(x, lengths), y, lengths, y_lengths).detach())
    assert final < initial * 0.45
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.model
def test_padding_does_not_change_valid_outputs():
    import torch

    from hanstream.model import CTCConfig, TinyCTC

    torch.manual_seed(7)
    model = TinyCTC(CTCConfig(4, 8, 4, 1)).eval()
    x = torch.randn(2, 7, 4)
    lengths = torch.tensor([4, 7])
    y = x.clone()
    y[0, 4:] = 100
    assert torch.allclose(model(x, lengths)[0, :4], model(y, lengths)[0, :4])


@pytest.mark.model
def test_repeated_target_feasibility():
    import torch

    from hanstream.model import ctc_loss

    with pytest.raises(ValueError, match='cannot be aligned'):
        ctc_loss(
            torch.zeros(1, 2, 3).log_softmax(-1),
            torch.tensor([1, 1]),
            torch.tensor([2]),
            torch.tensor([2]),
        )


@pytest.mark.model
def test_gradient_reaches_encoder():
    import torch

    from hanstream.model import CTCConfig, TinyCTC, ctc_loss

    model = TinyCTC(CTCConfig(3, 8, 4, 1))
    x = torch.randn(1, 6, 3)
    n = torch.tensor([6])
    ctc_loss(model(x, n), torch.tensor([1, 2]), n, torch.tensor([2])).backward()
    assert model.encoder.weight_ih_l0.grad.abs().sum() > 0
