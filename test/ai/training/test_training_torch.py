import pytest
import os
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch, MagicMock, PropertyMock

os.environ["SALTUP_BACKEND"] = "torch"
from saltup.saltup_env import SaltupEnv
SaltupEnv.SALTUP_BACKEND

import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from saltup.ai.classification.datagenerator import (
    ClassificationDataloader,
    keras_ClassificationDataGenerator,
    pytorch_ClassificationDataGenerator,
)
from saltup.ai.training.train import _train_model, training
from saltup.ai.training.callbacks import CallbackContext



@pytest.fixture
def mock_test_data_dir(tmp_path):
    """Create a mock test data directory with class subfolders and temporary jpg images."""
    class_names = ["class_0", "class_1"]
    for class_name in class_names:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(5):  # Create 2 images per class
            img_path = class_dir / f"image_{i}.jpg"
            # Generate a random image matrix and save it as an image
            random_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(random_image)
            image.save(img_path)
    return str(tmp_path)


class PyTorchModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PyTorchModel, self).__init__()
        self.fc = nn.Linear(32 * 32 * 3, num_classes)  # Fully connected layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.softmax(self.fc(x), dim=1)  # Apply softmax activation
        return x


@pytest.fixture
def mock_pytorch_model():
    """Create a mock PyTorch model."""
    return PyTorchModel(num_classes=2)


@pytest.fixture
def mock_pytorch_data_generator(mock_test_data_dir):
    """Create a mock PyTorch data generator."""
    class_dict = {"class_0": 0, "class_1": 1}
    dataloader = ClassificationDataloader(
        source=mock_test_data_dir, 
        classes_dict=class_dict, 
        img_size=(32, 32, 3)
    )
    return pytorch_ClassificationDataGenerator(
        dataloader=dataloader,
        target_size=(32, 32),
        num_classes=2,
        batch_size=4
    )

class TestTrainPytorch:
    """Test the PyTorch training function."""
    def test_training_pytorch_missing_optimizer(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test that training fails when optimizer is missing for PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        loss_function = nn.CrossEntropyLoss()
        
        with pytest.raises(ValueError, match="both `loss_function` and `optimizer` must be provided"):
            training(
                train_DataGenerator=mock_pytorch_data_generator,
                model=mock_pytorch_model,
                loss_function=loss_function,
                optimizer=None,
                epochs=1,
                output_dir=output_dir,
                kfold_param={'enable': False}
            )

    def test_train_model_pytorch(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test training a PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # The new implementation expects PyTorch DataLoader directly
            # Convert our data generator to DataLoader format
            train_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=True)
            val_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=False)
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model
            trained_model_path = _train_model(
                model=mock_pytorch_model,
                train_gen=train_loader,
                val_gen=val_loader,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=None,
                model_output_name="test_model"
            )
            
            # Assertions
            assert os.path.exists(trained_model_path)
            assert trained_model_path.endswith(".pth")
            assert os.path.exists(os.path.join(output_dir, "saved_models"))
            
    def test_training_without_kfold_pytorch(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test training without k-fold cross validation for PyTorch model."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose, \
             patch('saltup.ai.training.train.convert_torch_to_onnx') as mock_torch_onnx_conv:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model
            result = training(
                train_DataGenerator=mock_pytorch_data_generator,
                model=mock_pytorch_model,
                loss_function=loss_function,
                optimizer=optimizer,
                epochs=1,
                output_dir=output_dir,
                validation=[0.8, 0.2],
                kfold_param={'enable': False},
                model_output_name="test_model"
            )
            
            # Assertions
            assert result['kfolds'] is False
            assert len(result['models_paths']) >= 1
            assert os.path.exists(os.path.join(output_dir, "options.txt"))

class TestCallbackIntegration:
    """Test callback integration in training."""
    
    def test_pytorch_training_with_callbacks(self, mock_pytorch_model, mock_pytorch_data_generator, tmp_path):
        """Test PyTorch training with callbacks."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        with patch.object(type(SaltupEnv), 'SALTUP_TRAINING_PYTORCH_ARGS', new_callable=PropertyMock) as mock_pytorch_args, \
             patch.object(type(SaltupEnv), 'SALTUP_PYTORCH_DEVICE', new_callable=PropertyMock) as mock_device, \
             patch.object(type(SaltupEnv), 'SALTUP_KERAS_TRAIN_VERBOSE', new_callable=PropertyMock) as mock_verbose:
            
            mock_pytorch_args.return_value = {
                'early_stopping_patience': 0,
                'use_scheduler_per_epoch': False
            }
            mock_device.return_value = 'cpu'
            mock_verbose.return_value = 0
            
            # Create a mock callback
            mock_callback = Mock()
            mock_callback.on_train_begin = Mock()
            mock_callback.on_epoch_end = Mock()
            mock_callback.on_train_end = Mock()
            
            # Convert to DataLoader
            train_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=True)
            val_loader = DataLoader(mock_pytorch_data_generator, batch_size=4, shuffle=False)
            
            # Define loss function and optimizer
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mock_pytorch_model.parameters(), lr=0.001)
            
            # Train the model with callbacks
            trained_model_path = _train_model(
                model=mock_pytorch_model,
                train_gen=train_loader,
                val_gen=val_loader,
                output_dir=output_dir,
                epochs=1,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=None,
                model_output_name="test_model",
                app_callbacks=[mock_callback]
            )
            
            # Assertions
            assert os.path.exists(trained_model_path)
            mock_callback.on_train_begin.assert_called_once()
            mock_callback.on_epoch_end.assert_called_once()
            mock_callback.on_train_end.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])