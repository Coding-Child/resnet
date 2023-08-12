import torch
import torch.nn as nn
import torch.optim as opt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as T

from Dataset.pathology_dataset import PathologyDataset
from Trainer.trainer import seed_everything, main
from model.model import ResNet50

if __name__ == '__main__':
    seed_everything(42)

    lr = 1e-4
    weight_decay_lambda = 1e-4
    num_epochs = 200
    batch_size = 128
    accumulation_steps = 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNet50(num_classes=1)
    model.to(device)

    weight_class_1 = 3504 / (2838 * 2)
    pos_weight = torch.tensor([weight_class_1]).to(device)

    optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=weight_decay_lambda)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, patience=int(num_epochs * 0.1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    transform = T.Compose([T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip(),
                           T.RandomRotation(180),
                           T.ToTensor()
                           ])

    pathology_data = PathologyDataset(root_dir='data',
                                      transform=transform
                                      )

    data_size = len(pathology_data)
    train_size = int(data_size * 0.7)
    dev_size = int(data_size * 0.1)
    test_size = data_size - train_size - dev_size

    train_dataset, dev_dataset, test_dataset = random_split(pathology_data, [train_size, dev_size, test_size])

    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(dev_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    main(num_epochs=num_epochs,
         device=device,
         model=model,
         train_loader=train_loader,
         dev_loader=dev_loader,
         test_loader=test_loader,
         criterion=criterion,
         optimizer=optimizer,
         save_path='model_check_point',
         accumulation_steps=accumulation_steps,
         scheduler_epoch=scheduler
         )
