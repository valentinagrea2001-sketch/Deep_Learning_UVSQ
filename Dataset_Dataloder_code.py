import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import MNIST
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class MNISTDataset(Dataset):
    def __init__(self):
        # Comme MNIST est petit, on charge quand même tout en mémoire
        # Mais on ne convertit pas en float ni en NumPy, ce qui permet
        # quand même de diviser l'utilisation de mémoire ~ par 5
        data = MNIST('mnist', download=True)
        # On sauvegarde ce dont on a besoin en "self".
        self.images, self.labels = data.data, data.targets
        
    def __len__(self):
        # Ceci ne doit retourner que la longueur du dataset complet
        return len(self.images)
    
    def __getitem__(self, idx: int):
        # Ceci doit retourner l'élément idx-ième du dataset
        # Pas besoin de randomiser ici
        image = self.images[idx].float().view(28*28) / 255.0
        label = self.labels[idx]
        # Ça peut retourner n'importe quel objet python : tuple, dict, etc.
        return image, label


dataset = MNISTDataset()

def collate_fn(batch):
    # batch : une liste de longueur 32 = batch_size, avec batch[0] = premiere_image, premiere_etiquette, etc.
    images = []
    labels = []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    # Deux listes : images et labels, chacune de longueur 32 de éléments : torch.Tensor
    # stack "empile" les 32 tenseurs dans la dimension 0
    images = torch.stack(images, dim=0)  # taille : (32, 784)
    labels = torch.tensor(labels)  # taille : (32,)
    return images, labels

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=collate_fn
)


class TorchMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        
        all_dims = [input_dim] + list(hidden_dims) + [num_classes]
        
        layers = []
        for dim1, dim2 in zip(all_dims[:-1], all_dims[1:]):
            layers.append(torch.nn.Linear(dim1, dim2))
            layers.append(torch.nn.ReLU())
            
        # Il ne faut pas mettre de ReLU à la fin !
        layers = layers[:-1]
        
        # nn.Sequential crée un modèle qui appelle toutes 
        # les couches l'une après l'autre.
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


modele = TorchMLP(784, [100, 200, 100], 10)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modele.parameters(), lr=0.002)


valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=32,
    # Avec shuffle=False, on a les éléments de validation
    # toujours dans le même ordre ; cela ne change pas l'accuracy,
    # mais c'est utile par exemple si on veut toujours visualiser
    # les premiers points de validation, etc.
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn,
    drop_last=False
)

for epoch in range(10):
    # Tout le tour du jeu d'entraînement
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        y_pred = modele(images)
        loss_result = loss_function(y_pred, labels)
        loss_result.backward()
        optimizer.step()
    # On fait la prévision sur le jeu de validation
    # et on sauvegarde les résultats
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valid_dataloader:
            logits = modele(images)
            all_preds.extend(torch.argmax(logits, dim=1))
            all_labels.extend(labels)
    print(accuracy_score(all_labels, all_preds))
