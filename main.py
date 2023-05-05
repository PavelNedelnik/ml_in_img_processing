import numpy as np
import torch
import torchio as tio
import gc
from pathlib import Path
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
from src.dataset import BraTS21
from src.model import nnUNet
from src.loss import DiceBCELoss, dice_loss
import gc

images = Path('.') / 'images'

transforms = tio.Compose({
    tio.transforms.ZNormalization() : 1,
    tio.transforms.RandomAffine(scales=(0.7, 1.4)) : 0.2,
    tio.transforms.RandomAffine(degrees=(-30, 30)) : 0.2,
    tio.transforms.RandomElasticDeformation() : 0.2,
    tio.transforms.RandomGamma(log_gamma=(0.7, 1.5)) : 0.15
})
indices = np.arange(len(list(images.glob('BraTS2021*'))))
df = BraTS21(images, indices=indices, x_transforms=transforms)

def train_step(model, train_loader, optimizer, loss_fn):
    avg_loss = 0.

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        inputs, labels = inputs.to('cuda'), tuple(l.to('cuda') for l in labels)

        outputs = model.to('cuda')(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        avg_loss += loss.item()

        del loss, outputs, inputs, labels
        torch.cuda.empty_cache()
        _ = gc.collect()
        

    avg_loss = avg_loss / len(train_loader)

    return avg_loss

def fit():
    folds = list(random_split(df, [.2] * 5))
    for f in range(len(folds)):
        model = nnUNet().to('cuda')
        loss_fn = DiceBCELoss()
        scorer = dice_loss
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, nesterov=True)
        epochs = 1000
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=.9)
        batch_size = 4

        train_folds = folds[:f]
        if f + 1 < len(folds):
            train_folds += folds[f + 1:]

        train_loader = DataLoader(ConcatDataset(train_folds), batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(folds[f], batch_size=batch_size, shuffle=True)
        snapshot_num = 0
        for epoch in range(epochs):
            print('EPOCH {}:'.format(epoch + 1))

            model.train(True)
            avg_loss = train_step(model, train_loader, optimizer, loss_fn)
            scheduler.step()

            with torch.no_grad():
              running_vloss = [0, 0, 0]
              for i, (vinputs, vlabels) in enumerate(valid_loader):
                  vinputs, vlabels = vinputs.to('cuda'), tuple(l.to('cuda') for l in vlabels)
                  voutput = model(vinputs)[0]
                  for channel in range(3):
                      running_vloss[channel] += scorer(voutput[:, channel, :, :, :], vlabels[channel])
                  del voutput, vinputs, vlabels
                  torch.cuda.empty_cache()
                  _ = gc.collect()

              
              running_vloss = [l / (i + 1) for l in running_vloss]
                  
              print('LOSS train {} valid {}'.format(avg_loss, running_vloss))
              if epoch and not epoch % 100:
                torch.save(model.state_dict(), f'fold_{f}_model_snapshot_{snapshot_num}.pt')
                snapshot_num += 1


if __name__=='__main__':
    fit()