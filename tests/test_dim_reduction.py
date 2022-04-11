import os
import numpy as np
import nlp.dim_reduction as ae
from sklearn import datasets
import pytest


@pytest.fixture(scope='module')
def features():
    x, _ = datasets.make_regression(n_samples=100, n_features=100)
    return x


@pytest.mark.usefixtures("clean_dir")
class TestAutoEncoders:
    save_checkpoint = 'aemodels/last-checkpoint.bin'
    save_best_checkpoint = 'aemodels/best-checkpoint.bin'
    log_file = 'log.txt'
    lr = 1e-3
    n_epochs = 5
    batch_size = 20
    expected_class_attributes = ['fitter', 'history']

    def check_class_attribute(self, autoencoder: ae.Autoencoder):
        # Test main autoencoder attributes
        attributes = []
        for expected_attribute in self.expected_class_attributes:
            assert hasattr(autoencoder, expected_attribute)

    def check_model_files(self):
        # Test encoder files
        expected_files = [self.save_checkpoint, self.save_best_checkpoint, self.log_file]
        for expected_file in expected_files:
            assert os.path.isfile(expected_file)

    def check_training_components(self, autoencoder: ae.Autoencoder):
        # Test encoder model exist and is linked to father class
        assert hasattr(autoencoder.fitter, 'model')
        assert isinstance(autoencoder.fitter.model, ae._AEModel)
        # Test model was trained in n epochs
        assert len(autoencoder.history) == self.n_epochs
        # Test model learning rate was used
        assert self.lr in autoencoder.history['lr'].values

    @staticmethod
    def check_inference_results(predictions, features, layers):
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(features), layers[-1])

    def test_fit_and_transform(self, features):

        layers = [50, 10]
        autoencoder = ae.Autoencoder(layers=layers,
                                     batch_size=self.batch_size,
                                     lr=self.lr,
                                     n_epochs=self.n_epochs,
                                     verbose=False,
                                     save_checkpoint=self.save_checkpoint,
                                     save_best_checkpoint=self.save_best_checkpoint,
                                     )

        autoencoder.fit(features)
        x_hat = autoencoder.transform(features)
        # Test encoded shape
        self.check_inference_results(predictions=x_hat, features=features, layers=layers)
        # Test ae generate the expected files
        self.check_model_files()
        # Test ae contains the expected attributes
        self.check_class_attribute(autoencoder=autoencoder)
        # Check train components
        self.check_training_components(autoencoder=autoencoder)

    def test_fit_transform(self, features):
        # Test fit transform method
        layers = [50, 25, 5]
        autoencoder = ae.Autoencoder(layers=layers,
                                     batch_size=self.batch_size,
                                     lr=self.lr,
                                     n_epochs=self.n_epochs,
                                     verbose=False,
                                     save_checkpoint=self.save_checkpoint,
                                     save_best_checkpoint=self.save_best_checkpoint,
                                     )
        x_hat = autoencoder.fit_transform(features)
        # Test encoded shape
        self.check_inference_results(predictions=x_hat, features=features, layers=layers)
        # Test ae generate the expected files
        self.check_model_files()
        # Test ae contains the expected attributes
        self.check_class_attribute(autoencoder=autoencoder)
        # Check train components
        self.check_training_components(autoencoder=autoencoder)

