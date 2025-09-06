import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataset import DataSet
from dataloader.split import Split
from model.HarmonicNet import Statistics
from model.model import HarmonicCNN
from model_rnn.model import HarmonicRNN
from settings import Model
from settings import Settings as s


def evaluate(
    model_path: str | None, dataset: Split
) -> tuple[float | torch.Tensor, dict[str, float]]:

    device = torch.device(s.device)
    print(f"Evaluating on {device}")

    model = (HarmonicCNN() if s.model == Model.CNN else HarmonicRNN()).to(device)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded pre-trained model from {model_path}")
    else:
        raise ValueError("No model found insert a valid model")

    model.eval()

    test_dataset = DataSet(dataset, s.seconds)
    test_loader = DataLoader(test_dataset, batch_size=s.batch_size, shuffle=False)

    running_loss = 0.0
    total_batches = len(test_loader)

    stats = Statistics(0, 0, 0, 0, 0, 0)

    with torch.no_grad():
        for _, batch in tqdm(enumerate(test_loader), total=total_batches):
            model.set_input(batch, device)
            model.exec_forward()
            running_loss += model.get_loss()
            stats += model.get_batch_statistics()

    avg_loss = running_loss / total_batches

    return avg_loss, {
        "f1": stats.f1,
        "precision": stats.precision,
        "recall": stats.recall,
        "f1_bins": stats.f1_bins,
        "precision_bins": stats.precision_bins,
        "recall_bins": stats.recall_bins,
    }
