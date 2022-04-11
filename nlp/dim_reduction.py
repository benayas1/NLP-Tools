import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List
from .pt.fitter import AutoencoderFitter
from sklearn.base import BaseEstimator


class _AEDataset(torch.utils.data.Dataset):
    """
    Dataset Class providing data to the autoencoder
    """
    def __init__(self, data, device):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        x = self.data[key]
        item = {'x': x}
        return item


class _AEModel(torch.nn.Module):
    """
    Autoencoder model using PyTorch
    """
    def __init__(self, features, layers):
        super().__init__()

        self.encoder_layers, self.decoder_layers = [], []

        # encoder
        n_units = features
        for layer in layers:
            self.encoder_layers.append(torch.nn.Linear(in_features=n_units, out_features=layer))
            self.encoder_layers.append(torch.nn.ReLU())
            self.encoder_layers.append(torch.nn.BatchNorm1d(layer))
            self.encoder_layers.append(torch.nn.Dropout(0.2))
            n_units = layer
        self.encoder_layers = torch.nn.Sequential(*self.encoder_layers)

        # decoder
        for layer in reversed([features]+layers[:-1]):
            self.decoder_layers.append(torch.nn.Linear(in_features=n_units, out_features=layer))
            self.decoder_layers.append(torch.nn.ReLU())
            self.decoder_layers.append(torch.nn.BatchNorm1d(layer))
            self.decoder_layers.append(torch.nn.Dropout(0.2))
            n_units = layer
        self.decoder_layers = torch.nn.Sequential(*self.decoder_layers)

    def encoder(self, x):
        return self.encoder_layers(x)

    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.decoder_layers(x)
        return x


class Autoencoder(BaseEstimator):
    def __init__(self,
                 layers: List[int] = [],
                 device: str = None,
                 batch_size: int = 64,
                 lr: float = 1e-3,
                 n_epochs: int = 50,
                 early_stopping: int = 8,
                 verbose: bool = True,
                 save_checkpoint: str = None,
                 save_best_checkpoint: str = None,
                 ):
        """

        Args:
            layers (List[int]): A list containing the number of units in each layer.
                                The last element is the autoencoder bottleneck size.
            device (str): torch device. Can be 'cpu' or 'cuda'.
            batch_size (int, optional): Batch size to train the autoencoder. Defaults to 64.
            lr (float, optional): Learning rate to train the autoencoder. Defaults to 1e-3.
            epochs (int, optional): Number of epochs to train the autoencoder. Defaults to 50.
            early_stopping (int, optional): Early stopping value. Defaults to 8.
            verbose (bool, optional): Whether to print the autoencoder output or not. Defaults to True.
        """

        self.layers = layers
        self.device = device if device is not None else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.save_checkpoint = save_checkpoint
        self.save_best_checkpoint = save_best_checkpoint
        self.fitter = None
        self.history = None

    def fit(self, X):
        """
        Trains an autoencoder model

        Args:
            X (np.ndarray): The features matrix with shape (n_samples, n_features)

        Returns:
            self: pointer to self
        """

        train_loader = torch.utils.data.DataLoader(_AEDataset(X, self.device),
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # get the computation device
        model = _AEModel(X.shape[1], self.layers)
        model.to(self.device)

        # Define loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Create Fitter object
        self.fitter = AutoencoderFitter(model,
                                        self.device,
                                        loss=criterion,
                                        optimizer=optimizer,
                                        verbose=self.verbose,
                                        log_path='log.txt',
                                        )

        # Fit the model
        self.history = self.fitter.fit(train_loader,
                                       val_loader='training',
                                       n_epochs=self.n_epochs,
                                       early_stopping=self.early_stopping,
                                       early_stopping_mode='min',
                                       save_checkpoint=self.save_checkpoint,
                                       save_best_checkpoint=self.save_best_checkpoint,
                                       verbose_steps=50 if self.verbose is True else 0)

        # Load best performing model
        # if self.save_checkpoint is True:
        if self.save_checkpoint is not None:
            self.fitter.load(self.save_best_checkpoint)

        return self

    def transform(self, X):
        """
        Encodes sets of features using the autoencoder

        Args:
            X (np.ndarray): The features matrix with shape (n_samples, n_features)

        Returns:
            np.ndarray: The representation vector
        """
        # Encode vectors
        model = self.fitter.model.to(self.device)
        model.eval()
        new_vectors = []
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(X)), batch_size=self.batch_size)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                encoded_vectors = model.encoder(batch[0].to(self.device)).cpu().numpy()
                new_vectors.append(encoded_vectors)

        new_vectors = np.concatenate(new_vectors)

        return new_vectors

    def fit_transform(self, X):
        """
        Trains an autoencoder on a sets of features and then encode the same features

        Args:
            X (np.ndarray): The features matrix with shape (n_samples, n_features)

        Returns:
            np.ndarray: The representation vector
        """
        return self.fit(X).transform(X)
