from data_helper import TagData
from model import SequenceTaggingTransformer
from dataset import loader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from utils import AccuracyCounter, logging
from torch.nn.utils.rnn import pad_sequence

import torch

def train(model, ignore_label, loss_func, epochs, optimizer, train_loader, eval_loader, device=None):
    device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt_str = str(optimizer).replace('\n ', ',')
    logging(f'Training - loss:{loss_func}, epochs:{epochs}, optimizer:{opt_str}, device:{device}')
    for epoch in range(epochs):
        # Train
        model.train()
        avg_loss = None
        train_accuracy = AccuracyCounter(ignore_label)
        for i, (words, tags) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(words)
            out = out.view(-1, out.shape[-1])
            tags = pad_sequence(tags, padding_value=ignore_label).to(device).view(-1)
            loss = loss_func(out, tags)
            avg_loss = loss.detach().item() if avg_loss is None else (0.99 * avg_loss + 0.01 * loss.detach().item())
            train_accuracy.compute_from_soft(out.detach(), tags.detach())
            loss.backward()
            optimizer.step()

        train_accuracy_val = train_accuracy.get_accuracy_and_reset()
        # Eval
        model.eval()
        with torch.no_grad():
            eval_accuracy = AccuracyCounter(ignore_label)
            for words, tags in eval_loader:
                out = model(words)
                out = out.view(-1, out.shape[-1])
                tags = pad_sequence(tags, padding_value=ignore_label).to(device).view(-1)
                eval_accuracy.compute_from_soft(out.detach(), tags.detach())
            eval_accuracy_val = eval_accuracy.get_accuracy_and_reset()
            logging('Done epoch {}/{} ({} batches) train accuracy {:.2f}, eval accuracy {:.2f} avg loss {:.5f}'.format(
                epoch+1, epochs, (epoch+1)*train_loader.__len__(), train_accuracy_val, eval_accuracy_val, avg_loss))




if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging(f'Device For Training:\'{device}')
    data = TagData('pos')
    model = SequenceTaggingTransformer(len(data.words), len(data.tags), 128, 8, 1024, 8, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=0.001,  betas=(0.9, 0.98), eps=1e-09)
    ignore_label = len(data.tags)
    train(model, ignore_label, CrossEntropyLoss(ignore_index=ignore_label, reduction='mean'), 10, optimizer, loader(data, 'train', 50), loader(data, 'dev', 50))
