# ======================================================================== #
# >>> Module with training, testing and reprocessing functions in pytorch. #                                        
# ======================================================================== #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
# Neural Network modeling:
# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Metrics:
# Torch Metrics
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryRecall, BinaryConfusionMatrix, BinaryPrecision, BinaryNegativePredictiveValue, BinaryROC, BinaryPrecisionRecallCurve, BinaryAveragePrecision

# Hypertunning Pytorch:
# Optuna
import optuna

# Python:
# OS
import os
# Tempfile
import tempfile
# Warnings
import warnings
# Time
import time
# Graphics:
# Matplotlib
import matplotlib.pyplot as plt

# Other utils:
# Tqdm
from tqdm import tqdm
# Numpy
import numpy as np

# ======================================================== #
# PyTorch - Class                                          #
# ======================================================== #

class PyTorch:
    
    # ======================================================== #
    # Inicialize - Class                                       #
    # ======================================================== #
    def __init__(
        self, 
        name: str = 'PyTorch'
    ):
        self.name = name

    # ======================================================== #
    # Dataset Pytorch - Class                                  #
    # ======================================================== #
    class Dataset(torch.utils.data.Dataset):
        """
        Custom PyTorch Dataset class for handling tabular data with separate indices
        for categorical features, numerical features, and labels.

        This class enables flexible extraction of different feature types from
        a preprocessed dataset (e.g., NumPy array) and returns them as PyTorch tensors,
        which is useful for deep learning models that treat categorical and numerical
        features differently (e.g., when using embeddings).

        Attributes:
            data (array-like): The full dataset (e.g., NumPy array) containing all features and labels.
            cat_idx (list): A list with two elements [start, end] indicating the range of categorical columns.
            num_idx (list): A list with two elements [start, end] indicating the range of numerical columns.
            label_idx (list): A list with two elements [start, end] indicating the range of label columns.

        Methods:
            __len__(): Returns the total number of samples in the dataset.
            __getitem__(idx): Returns a tuple (categorical_data, numerical_data, labels) for a given index.
        """
        # Initializing Attributes
        def __init__(
            self, 
            dataset,
            cat_idx: list = [],
            num_idx: list = [],
            label_idx: list = [],
        ):
            try:

                # Loading data
                self.data = dataset
                self.cat_idx = cat_idx
                self.num_idx = num_idx
                self.label_idx = label_idx

            except Exception as e:
                print(f'[ERROR] Failed to load dataset: {str(e)}.')
        
        # Len function Torch
        def __len__(
            self,
        ):
            try:

                return len(self.data)  
                
            except Exception as e:
                print(f'[ERROR] Failed to len data: {str(e)}.')

        # Get item 
        def __getitem__(
            self,
            idx: int,
        ):
            try:

                # Split of categorical, numerical and label variables
                categorical_data = self.data[idx][self.cat_idx[0] : self.cat_idx[1]]
                numerical_data = self.data[idx][self.num_idx[0] : self.num_idx[1]]
                labels = self.data[idx][self.label_idx[0] :  self.label_idx[1]]

                # Transform to tensors
                categorical_data = torch.from_numpy(categorical_data.astype(np.int64))
                numerical_data = torch.from_numpy(numerical_data.astype(np.float32))
                labels = torch.from_numpy(labels.astype(np.float32))

                return categorical_data, numerical_data, labels
            
            except Exception as e:
                print(f'[ERROR] Failed to find the indices and transform the data into tensors: {str(e)}')

    # ======================================================== #
    # Neural NetWork - Class                                   #
    # ======================================================== #    
    class Net(nn.Module):
        """
        Neural network for binary classification using both categorical and numerical features.

        This architecture uses embedding layers for categorical features and dense layers 
        for numerical features. Both feature types are combined and passed through a 
        deep neural network with batch normalization, dropout, and LeakyReLU activations.
        """
        # Initializing network parameters and attributes
        def __init__(
            self, 
            l1: int = 256,
            l2: int = 128,
            l3: int = 64,
            dropout_rate: float = 0.5,
            classes_per_cat: list = [2, 4, 7, 6, 4],  # Number of classes per categorical variable
            num_numerical_features: int = 13,
            prior_minoritary_class: float = 0,
            negative_slope: float = 0.01
        ):
            """
            Initializes the network architecture and parameters.

            Args:
                l1 (int): Number of units in the first dense layer for numerical features.
                l2 (int): Number of units in the first combined layer (numerical + categorical).
                l3 (int): Number of units in the second combined layer.
                dropout_rate (float): Dropout rate applied after the first combined layer.
                classes_per_cat (list): List with the number of classes per categorical feature.
                num_numerical_features (int): Number of input numerical features.
                prior_minoritary_class (float): Proportion of the positive (minority) class; used to adjust output layer bias.
                negative_slope (float): Negative slope used in the LeakyReLU activation function.
            """
            try:

                # Running __init__ of the nn.Module class
                super().__init__()
                
                # Embedding dims
                embedding_dims = [int(min(50, np.sqrt(n))) for n in  classes_per_cat]

                # Creating embeddings dynamically
                self.embeddings = nn.ModuleList(
                    [nn.Embedding(num_embeddings, emb_dim) for num_embeddings, emb_dim in zip(classes_per_cat, embedding_dims)]
                )

                # Total dimensions of the embedding layers
                self.total_embedding_dim = sum(embedding_dims)

                # Layer for numeric variables
                self.numerical_layer = nn.Linear(num_numerical_features, l1)
                self.bn_num = nn.BatchNorm1d(l1)
                
                # Combined the layers
                # Layer 1
                self.combined_layer_1 = nn.Linear(self.total_embedding_dim + l1, l2)
                self.bn1 = nn.BatchNorm1d(l2)

                # Layer 2
                self.combined_layer_2 = nn.Linear(l2, l3, bias = False)
                self.bn2 = nn.BatchNorm1d(l3)

                # Dropout Layer
                self.dropout_layer = nn.Dropout(dropout_rate)

                # Output Layer
                self.output_layer = nn.Linear(l3, 1)
                
                # Activation LeakyReLU
                self.leaky_relu = nn.LeakyReLU(negative_slope)

                # Bias adjustment according to the probability of the positive (minority) class
                self.prior_minoritary_class = prior_minoritary_class

                # Initialization weights
                self._init_weights()
            
            except Exception as e:
                print(f'[ERRO] Failed to load network attributes: {str(e)}')
        
        # Initialization Weights
        def _init_weights(
            self,
        ):
            """
            Initializes the weights of the network layers.

            - Dense layers are initialized using Kaiming Normal initialization.
            - Embedding layers are initialized using Xavier Uniform initialization.
            - The output layer bias is set based on the prior probability of the minority class (if provided).
            """
            try:
                # Linear Layers
                for layer in [self.numerical_layer, self.combined_layer_1, self.combined_layer_2]:
                    
                    # Weights Linear Layers
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'leaky_relu', a = self.leaky_relu.negative_slope)
                        # Bias Linear Layers
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                
                # Embbedings
                for embedding in self.embeddings:
                    nn.init.xavier_uniform_(embedding.weight)
                
                # Special adjustment for output layer
                nn.init.xavier_uniform_(self.output_layer.weight)
                

                # Initializing the output layer bios for imbalanced data 
                # Replacing with the actual proportion of the positive class
                if self.prior_minoritary_class > 0:
                    eps = 1e-6
                    prior_adjusted = max(min(self.prior_minoritary_class, 1 - eps), eps)
                    nn.init.constant_(
                        self.output_layer.bias, 
                        torch.log(torch.tensor(prior_adjusted / (1 - prior_adjusted), dtype = torch.float))
                    )

                else:
                    nn.init.zeros_(self.output_layer.bias)
                
            except Exception as e:
                print(f'[ERROR] Failed to perform weight initialization: {str(e)}.')

        # Forward
        def forward(
            self,
            cat_data, 
            num_data
        ):
            """
            Forward pass of the neural network.

            Args:
                cat_data (torch.Tensor): Tensor of categorical feature indices.
                    Shape: (batch_size, num_categorical_features)
                num_data (torch.Tensor): Tensor of numerical feature values.
                    Shape: (batch_size, num_numerical_features)

            Returns:
                torch.Tensor: Logits output by the network.
                    Shape: (batch_size, 1)
            """
            try:

                # Processing categorical variables with embeddings
                embedded_features = [embedding(cat_data[:, i]) for i, embedding in enumerate(self.embeddings)]
                combined_embeddings = torch.cat(embedded_features, dim = 1)
                combined_embeddings = self.dropout_layer(combined_embeddings)

                # Processing of numerical variables
                numerical_out = self.numerical_layer(num_data)
                numerical_out = self.bn_num(numerical_out)
                numerical_out = self.leaky_relu(numerical_out)
                numerical_out = self.dropout_layer(numerical_out)

                # Combining embeddings with numerical data
                combined = torch.cat([numerical_out, combined_embeddings], dim = 1)

                # Passage through the neural network
                # combined_layer_1
                x = self.combined_layer_1(combined)
                x = self.bn1(x)
                x = self.leaky_relu(x)
                x = self.dropout_layer(x)

                # combined_layer_2
                x = self.combined_layer_2(x)
                x = self.bn2(x)
                x = self.leaky_relu(x)
              
                # Logits
                logits = self.output_layer(x)

                return logits
            
            except Exception as e:
                print(f'[ERROR] Failed to execute neural network forward: {str(e)}')

    # ======================================================== #
    # Focal Loss - Class                                       #
    # ======================================================== # 
    class FocalLoss(nn.Module):
        """
        Focal Loss for binary classification tasks.

        This loss function combats class imbalance by
        reducing the contribution of simple (already well-classified) examples
        and placing more emphasis on difficult examples through a focus factor.

        Formula:
        FL = - alpha_t * (1 - p_t)^gamma * log(p_t)

        Where:
        - p_t = p when y=1, and (1-p) when y=0
        - alpha_t is a class-specific weight
        - gamma >= 0 is the focus parameter
        """
        # Inicialize Class
        def __init__(
            self, 
            alpha: list = [1.0, 1.0], 
            gamma: float = 2.0, reduction: 
                str = 'mean'
        ):
            """
            Initializes FocalLoss.

            Args:
            alpha (list): Class weights [negative_alpha, positive_alpha].
            gamma (float): Focus parameter (γ).
            reduction (str): Reduction method ('none', 'mean', or 'sum').
            """
            try:
                
                super().__init__()
                self.gamma = gamma
                self.reduction = reduction

                # Store alpha as buffer (goes to GPU/CPU along with the model)
                self.register_buffer('alpha', torch.tensor(alpha, dtype = torch.float))

            except Exception as e:
                print(f'[ERROR] Failed load FocalLoss atributes: {str(e)}')

        def forward(
            self, 
            inputs, 
            targets
        ):
            """
            Calculates Focal Loss.

            Args:
            inputs (Tensor): Model logits. Shape (N,) or (N,1).
            targets (Tensor): Binary labels {0,1}. Shape (N,) or (N,1).

            Returns:
            Tensor: Scalar or sample loss, depending on the reduction.
            """
            try:

                # Adjust shape (guarantees 1D vector)
                inputs = inputs.view(-1)
                targets = targets.view(-1).float()

                # BCE (stable, works directly with logits)
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = 'none')

                # Probabilities (for the focal factor)
                probs = torch.sigmoid(inputs)

                # p_t = probability of the true class
                p_t = probs * targets + (1 - probs) * (1 - targets)

                # Focus factor (1 - p_t)^γ
                focal_factor = (1 - p_t).pow(self.gamma)

                # Alpha selection by class
                alpha_factor = self.alpha[1] * targets + self.alpha[0] * (1 - targets)

                # Final Loss
                loss = alpha_factor * focal_factor * bce_loss

                # Reduction
                if self.reduction == 'mean':
                    return loss.mean()
                elif self.reduction == 'sum':
                    return loss.sum()
                elif self.reduction == 'none':
                    return loss
                else:
                    raise ValueError(f"Invalid reduction mode: {self.reduction}. "
                                    f"Choose from 'none', 'mean', 'sum'.")
            except Exception as e:
                print(f'[ERROR] Failed to calculate focal loss: {str(e)}')

    # ======================================================== #
    # Early Stopping - Class                                   #
    # ======================================================== #     
    class EarlyStopping:
        
        """
        Implements early stopping to terminate training when a monitored metric stops improving.

        Attributes:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored score to qualify as an improvement.
            mode (str): One of ['min', 'max']. In 'min' mode, training stops when the score increases;
                in 'max' mode, training stops when the score decreases.
            save_path (str or Path): Path to save the best model.
            tempfile_save (bool): Whether to save using a temporary file.
            verbose (bool): If True, prints messages during training.
            best_score (float): Best score observed so far.
            counter (int): Number of epochs since the last improvement.
            early_stop (bool): Whether early stopping was triggered.
            best_epoch (int): Epoch number at which the best score was achieved.
        """
        # Initializing attributes
        def __init__(
            self,
            patience: int = 15,
            min_delta: float = 1e-4,
            mode: str = 'max',
            save_path = None,
            tempfile_save = False,
            verbose: bool = True,
        ):
            """
            Initializes the EarlyStopping object.

            Args:
                patience (int): Number of epochs to wait for improvement before stopping.
                min_delta (float): Minimum score improvement to reset the patience counter.
                mode (str): 'max' for maximizing the metric, 'min' for minimizing.
                save_path (str or Path): File path to save the best model.
                tempfile_save (bool): Use temporary file handling for saving the model.
                verbose (bool): If True, logs progress and stopping messages.
            """
            try:
                self.patience = patience
                self.min_delta = min_delta
                self.mode = mode
                self.save_path = save_path
                self.verbose = verbose
                self.tempfile_save = tempfile_save
                self.best_score = None
                self.counter = 0
                self.early_stop = False
                self.best_epoch = None
            except Exception as e:
                print(f'[ERROR] Failed to initialize attributes of Early Stopping class: {str(e)}.')
        
        # Call
        def __call__(
            self,
            epoch: int,
            score: float, 
            model = None,
        ):  
            """
            Evaluates whether the model has improved and saves the best model if applicable.

            Args:
                epoch (int): Current epoch number.
                score (float): The value of the monitored metric at the current epoch.
                model (torch.nn.Module, optional): The model to save if improvement is detected.
            """
            try:
                # If the model as been improved
                improved = False

                # First score
                if self.best_score is None:
                    improved = True

                # Max Mode
                elif self.mode == 'max' and score > self.best_score + self.min_delta:
                    improved = True
                
                # Min Mode
                elif self.mode == 'min' and score < self.best_score - self.min_delta:
                    improved = True
                
                # Improved Score
                if improved:
                    
                    self.best_score = score
                    self.counter = 0
                    self.early_stop = False
                    self.best_epoch = epoch if epoch is not None else 0

                    # Saving the model and the path
                    if model and self.save_path:
                        
                        # Tempfile
                        if self.tempfile_save:
                            model_to_save = model.module if isinstance(model, nn.DataParallel) else model 
                            torch.save(model_to_save.state_dict(), self.save_path.name)

                        # No Tempfile
                        else:
                            model_to_save = model.module if isinstance(model, nn.DataParallel) else model 
                            torch.save(model_to_save.state_dict(), self.save_path)
                        
                        if self.verbose:
                            print(f'✅ Model Improvement (Epoch: {self.best_epoch}, Score: {self.best_score:.5f})')
                
                # No Improvement
                else:
                    # Counter
                    self.counter += 1
                    if self.verbose:
                        print(f'⏳ EarlyStopping: {self.counter}/{self.patience} no improvement (Current Score: {score:.5f})')
                    
                    # Early Stopping
                    if self.counter >= self.patience:
                        self.early_stop = True
                        if self.verbose:
                            print(f'🛑 Stopping training by early stopping (no improvement after: {self.patience} epochs.)')
                            print(f"✅ Best Model saved in: '{self.save_path}' (Epoch: {self.best_epoch}, Score: {self.best_score:.5f}).")
            
            except Exception as e:
                print(f'[ERROR] Failed to execute Early stopping: {str(e)}')

        # Reset Atributes
        def reset(
            self
        ):
            """
            Resets the internal state of the EarlyStopping instance.
            """
            try:

                self.best_score = None
                self.counter = 0
                self.early_stop = False
                self.best_epoch = 0
            
            except Exception as e:
                print(f'[ERROR] Failed to reset Early Stooping class attributes: {str(e)}.')

    # ======================================================== #
    # PyTorch Flow - Class                                     #
    # ======================================================== #    
    class PyTorchFlow:

        """
        Wrapper class for managing the training pipeline of a PyTorch model,
        including data loading, architecture parameters, training configuration,
        early stopping, and evaluation with k-fold cross-validation.

        Attributes:
            trainset (Dataset): Training dataset.
            testset (Dataset): Test dataset.
            l1 (int): Number of units in the first hidden layer.
            l2 (int): Number of units in the second hidden layer.
            l3 (int): Number of units in the third hidden layer.
            dropout_rate (float): Dropout rate for regularization.
            num_workers (int): Number of subprocesses for data loading.
            batch_size (int): Number of samples per batch.
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for regularization.
            max_epochs (int): Maximum number of training epochs.
            early_stopping_p (int): Patience for early stopping.
            early_stopping_mode (str): Mode for early stopping, either 'min' or 'max'.
            save_path_model (str): File path to save the best model.
            seed (int): Random seed for reproducibility.
            k_fold (int): Number of folds for cross-validation.
            target_score (str): Metric used to determine model performance (e.g., 'roc').
        """
        # Initialize Atributes
        def __init__(
            self,
            trainset = None, 
            testset = None,
            l1: int = 256,
            l2: int = 128,
            l3: int = 64,
            dropout_rate: float = 0.5,
            num_workers: int = 2,
            batch_size: int = 128,
            lr: float = 1e-3,
            weight_decay: float = 1e-5,
            gamma: float = 2.0,
            alpha: list = [1.0, 1.0],
            max_epochs: int = 200,
            early_stopping_p: int = 15,
            early_stopping_mode: str = 'max',
            save_path_model: str = '/best_model.pt',
            seed: int = 33,
            k_fold: int = 5,
            target_score: str = 'roc',
            
        ):
            try:

                self.trainset = trainset
                self.testset = testset
                self.l1 = l1
                self.l2 = l2
                self.l3 = l3
                self.dropout_rate = dropout_rate
                self.num_workers = num_workers
                self.batch_size = batch_size
                self.lr = lr
                self.weight_decay = weight_decay
                self.gamma = gamma
                self.alpha = alpha
                self.max_epochs = max_epochs
                self.early_stopping_p = early_stopping_p
                self.early_stopping_mode = early_stopping_mode
                self.save_path_model = save_path_model
                self.seed = seed
                self.k_fold = k_fold
                self.target_score = target_score

            except Exception as e:
                print(f'[ERROR] Failed to load PytorchFlow class attributes {str(e)} .')

        # ======================================================== #
        # Device - Function                                        #
        # ======================================================== #
        def _device(
            self,
            net,
            device: str = None,
        ):
            """
            Moves the given PyTorch model to the appropriate device (CPU or GPU).
            If multiple GPUs are available, wraps the model using `nn.DataParallel`.

            Args:
                net (nn.Module): The PyTorch model to be moved.
                device (str, optional): Device identifier (e.g., 'cpu', 'cuda:0'). 
                    If None, it will be automatically selected based on availability.

            Returns:
                Tuple[nn.Module, str]: A tuple containing the model moved to the device 
                    and the device identifier string.

            Raises:
                Exception: If the model fails to move to the specified device.
            """
            try:

                # Automatically detect device if not provided
                if device is None:
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                # If more than one GPU is available, apply DataParallel
                if torch.cuda.device_count() > 1 and device.startswith('cuda'):
                    net = nn.DataParallel(net)
                
                # Move the model to the device
                net.to(device)

                return net, device
            except Exception as e:
                print(f'[ERROR] Failed to move network to device: {str(e)}.')

        # ======================================================== #
        # Create Kfolds - Function                                 #
        # ======================================================== #
        def _kfolds(
            self
        ):  
            """
            Splits the training dataset into k folds for cross-validation.

            This method creates `k_fold` random subsets (folds) of the training dataset, 
            ensuring reproducibility using a fixed random seed. The final fold may be slightly 
            larger if the dataset size is not perfectly divisible by `k_fold`.

            Returns:
                List[Subset]: A list of PyTorch `Subset` objects representing each fold.

            Raises:
                Exception: If the dataset fails to split into folds.
            """
            try:
                # Spliting folds for cross validation
                fold_size = len(self.trainset) // self.k_fold
                fold_sizes = [fold_size] * (self.k_fold - 1) + [len(self.trainset) - fold_size * (self.k_fold - 1)]
                folds = torch.utils.data.random_split(self.trainset, fold_sizes, generator = torch.Generator().manual_seed(self.seed))
                
                return folds

            except Exception as e:
                print(f'[ERROR] Failed to create k folds of trainset: {str(e)}.')
        
        # ======================================================== #
        # OverSampling - Function                                  #
        # ======================================================== #
        def _oversampling(
            self, 
            trainset,
        ):  
            """
            Applies oversampling to balance class distribution in the training dataset.

            This method calculates class weights based on the frequency of each class and creates 
            a `WeightedRandomSampler` to allow oversampling of the minority class during training. 
            It also returns the proportion of the positive class in the dataset.

            Args:
                trainset (Dataset): The training dataset, where each item returns a tuple 
                    (features, ..., label). The label must be the last element and compatible 
                    with `torch.int64`.

            Returns:
                Tuple[WeightedRandomSampler, float]: 
                    - A PyTorch `WeightedRandomSampler` to be used in a DataLoader.
                    - The proportion of the positive class (class 1) in the dataset.

            Raises:
                Exception: If there is an error while computing class weights or creating the sampler.
            """
            try:
                
                # Compute class weights for imbalance handling
                eps = 1e-6
                all_class = torch.cat([labels for _, _, labels in trainset]).to(torch.int64)
                class_counts = torch.bincount(all_class)
                total_samples = len(all_class)
                num_classes = len(class_counts)
                class_weights = total_samples / (class_counts.float() + eps)

                # Create Weighted Sampler
                sample_weights = class_weights[all_class]
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights = sample_weights, 
                    num_samples = len(sample_weights), 
                    replacement = True
                )
                # Calculate minority class proportion
                prop_class_positive = class_counts[1].item() / total_samples

                return sampler, prop_class_positive
            
            except Exception as e:
                print(f'[ERROR] Failed to create sampler and calculate class distributions: {str(e)}.')

        # ======================================================== #
        # Plot Metrics - Function                                  #
        # ======================================================== #
        def _plot_metrics(
            self,
            avg_loss_t: list = None,
            avg_loss_v: list = None,
            avg_accuracy_t: list = None,
            avg_accuracy_v: list = None,
            avg_precision_t: list = None, 
            avg_precision_v: list = None,
            avg_npv_t: list = None, 
            avg_npv_v: list = None,
            avg_recall_t: list = None, 
            avg_recall_v: list = None,
            avg_auc_t: list = None, 
            avg_auc_v: list = None
        ):
            """
            Plots training and validation metrics over epochs for model convergence visualization.

            Args:
                avg_loss_t (list, optional): List of average training loss values per epoch.
                avg_loss_v (list, optional): List of average validation loss values per epoch.
                avg_accuracy_t (list, optional): List of average training accuracy values per epoch.
                avg_accuracy_v (list, optional): List of average validation accuracy values per epoch.
                avg_precision_t (list, optional): List of average training precision values per epoch.
                avg_precision_v (list, optional): List of average validation precision values per epoch.
                avg_npv_t (list, optional): List of average training negative predictive value per epoch.
                avg_npv_v (list, optional): List of average validation negative predictive value per epoch.
                avg_recall_t (list, optional): List of average training recall values per epoch.
                avg_recall_v (list, optional): List of average validation recall values per epoch.
                avg_auc_t (list, optional): List of average training AUC-ROC values per epoch.
                avg_auc_v (list, optional): List of average validation AUC-ROC values per epoch.

            Raises:
                Exception: If an error occurs during plotting.

            """
            try:
                # List of metrics
                metrics, train_metrics, val_metrics = [], [], []

                # Dynamically grouping metrics
                if avg_loss_t and avg_loss_v:
                    metrics.append('Loss')
                    train_metrics.append(avg_loss_t)
                    val_metrics.append(avg_loss_v)

                if avg_accuracy_t and avg_accuracy_v:
                    metrics.append('Accuracy')
                    train_metrics.append(avg_accuracy_t)
                    val_metrics.append(avg_accuracy_v)

                if avg_precision_t and avg_precision_v :
                    metrics.append('Precision')
                    train_metrics.append(avg_precision_t)
                    val_metrics.append(avg_precision_v)

                if avg_npv_t and avg_npv_v:
                    metrics.append('NPV')
                    train_metrics.append(avg_npv_t)
                    val_metrics.append(avg_npv_v)

                if avg_recall_t and avg_recall_v:
                    metrics.append('Recall')
                    train_metrics.append(avg_recall_t)
                    val_metrics.append(avg_recall_v)

                if avg_auc_t and avg_auc_v:
                    metrics.append('AUC-ROC')
                    train_metrics.append(avg_auc_t)
                    val_metrics.append(avg_auc_v)

                # Total number of metrics
                num_metrics = len(train_metrics)

                num_cols = 3
                num_rows = (num_metrics + num_cols - 1) // num_cols

                plt.rc('font', size = 10)
                fig, axes = plt.subplots(num_rows, num_cols, figsize = (6 * num_cols, 4 * num_rows))
                axes = axes.flatten() if num_metrics > 1 else [axes]

                for i in range(num_metrics):
                    ax = axes[i]
                    ax.plot(train_metrics[i], label = 'Training')
                    ax.plot(val_metrics[i], label = 'Validation')
                    ax.set_title(f'Model Convergence - {metrics[i]}')
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel(metrics[i])
                    ax.legend()
                    ax.grid(True, alpha = 0.6, linestyle = 'dotted')

                # Remove extra subplots, if any
                for j in range(num_metrics, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                plt.show()
            
            except Exception as e:
                print(f'[ERROR] Failed to generate graphs of metrics during epochs: {str(e)}.')

        # ======================================================== #
        # Confusion Matrix - Function                              #
        # ======================================================== #
        def _confusion_matrix(
            self,
            preds,
            labels,
            title: str = 'Confusion Matrix',
        ):
            """
            Plots the confusion matrix for binary classification predictions.

            Args:
                preds (Tensor or array-like): Predicted labels or probabilities from the model.
                labels (Tensor or array-like): True labels corresponding to the predictions.
                title (str, optional): Title for the confusion matrix plot. Default is 'Confusion Matrix'.

            Raises:
                Exception: If an error occurs during the plotting process.

            """
            try:

                # Confusion Matrix
                plt.rc('font', size = 10)
                fig, ax= plt.subplots(figsize = (8, 4))
                ax.grid(False)
                
                metric = BinaryConfusionMatrix()
                metric(preds, labels)
                fig_, ax_ = metric.plot(cmap = 'viridis', ax = ax)
                ax_.set_title(title)
                plt.show()
    
            except Exception as e:
                print(f'[ERROR] failed to plot confusion matrix: {str(e)}.')

        # ======================================================== #
        # Flow Cross Validation - Function                         #
        # ======================================================== #
        def CrossValidation(
            self,
        ):
            """
            Perform k-fold cross-validation training and evaluation of the PyTorch model.

            This method performs the following steps for each fold:
            - Splits the training data into training and validation sets.
            - Applies oversampling to handle class imbalance.
            - Creates data loaders for training and validation.
            - Initializes the neural network, loss function, optimizer, and learning rate scheduler.
            - Trains the model for a maximum number of epochs or until early stopping triggers.
            - Evaluates the model on validation data after training.
            - Collects and prints various metrics (Loss, Accuracy, Precision, NPV, Recall, AUC-ROC).
            - Plots training and validation metric curves and the confusion matrix.
            - Aggregates metrics across folds and prints mean and standard deviation.

            Attributes used:
                self.trainset: Dataset used for training and cross-validation splits.
                self.k_fold (int): Number of folds for cross-validation.
                self.batch_size (int): Batch size for data loaders.
                self.num_workers (int): Number of workers for data loading.
                self.l1, self.l2, self.l3 (int): Neural network layer sizes.
                self.dropout_rate (float): Dropout rate in the network.
                self.lr (float): Learning rate for optimizer.
                self.weight_decay (float): Weight decay for optimizer.
                self.max_epochs (int): Maximum number of epochs for training.
                self.early_stopping_p (int): Patience parameter for early stopping.
                self.early_stopping_mode (str): Mode ('max' or 'min') for early stopping.
                self.save_path_model (str): File path to save the best model.
                self.target_score (str): Metric used for early stopping ('roc', 'accuracy', etc.).

            Raises:
                Exception: Prints error message if any exception occurs during the cross-validation process.

            """
            try:

                folds = self._kfolds()
                # Defining list to store fold metrics
                accuracy_kfolds, recall_kfolds, auc_kfolds, loss_kfolds, precision_kfolds, npv_kfolds = [], [], [], [], [], []

                # Initialize Metrics for binary classification

                # Training
                accuracy_train = BinaryAccuracy()
                recall_train = BinaryRecall()
                auc_train = BinaryAUROC(thresholds = None)
                precision_train = BinaryPrecision()
                npv_train = BinaryNegativePredictiveValue()

                # Validation
                accuracy_val = BinaryAccuracy()
                recall_val = BinaryRecall()
                auc_val = BinaryAUROC(thresholds = None)
                precision_val = BinaryPrecision()
                npv_val = BinaryNegativePredictiveValue()

                # Score target
                if self.target_score == 'accuracy':
                    target_score = BinaryAccuracy()
                
                elif self.target_score == 'recall':
                    target_score = BinaryRecall()
                
                elif self.target_score == 'roc':
                    target_score = BinaryAUROC(thresholds = None)
                
                elif self.target_score == 'precision':
                    target_score = BinaryPrecision()

                elif self.target_score == 'npv':
                    target_score = BinaryNegativePredictiveValue()

                for i in tqdm(range(self.k_fold), desc = '\nCross - Validation Progress', leave = False):

                    # Separating training and validation data
                    # Fold for validation
                    val_set = folds[i] 

                    # All training folds except validation fold
                    train_sets = [folds[j] for j in range(self.k_fold) if j != i]
                    train_set = torch.utils.data.ConcatDataset(train_sets)

                    sampler, prop_class_positive =  self._oversampling(trainset = train_set)
                    
                    # Train Loader
                    trainloader = torch.utils.data.DataLoader(
                        train_set, 
                        batch_size = self.batch_size, 
                        sampler = sampler, 
                        num_workers = self.num_workers,
                        drop_last = True,    
                    )

                    # Val Loader
                    valloader = torch.utils.data.DataLoader(
                        val_set, 
                        batch_size = self.batch_size, 
                        shuffle = False, 
                        num_workers = self.num_workers, 
                        drop_last = False,
                    )
                    

            
                    # Loading Net
                    net = PyTorch.Net(
                        l1 = self.l1,
                        l2 = self.l2, 
                        l3 = self.l3, 
                        dropout_rate = self.dropout_rate,
                        prior_minoritary_class = prop_class_positive,  
                    )
                    # Moving the network to the device
                    net, device = self._device(net)

                    # Criterion
                    criterion = PyTorch().FocalLoss(alpha = self.alpha, gamma = self.gamma).to(device)

                    # Optimizer
                    optimizer = optim.AdamW(net.parameters(), lr = self.lr, weight_decay = self.weight_decay) 

                    # Scheduler
                    # Warmup (linear from 1e-5 to 0.001)
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor = 0.01,
                        total_iters = int(self.max_epochs * 0.05),
                    )
                    # Cosine Annealing after warmup
                    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max = self.max_epochs - int(self.max_epochs * 0.05),
                        eta_min = 1e-6,
                    )
                    # Composition: 10 warmup epochs + (max_epochs cosine - 10 warmup)
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers = [warmup_scheduler, cosine_scheduler],
                        milestones = [int(self.max_epochs * 0.05)],
                    )

                    # Adjusting error caused by scheduler
                    warnings.filterwarnings('ignore', category = UserWarning)

                    # Info Cross-Validation
                    print(f'\n\nTraining fold sample set:')
                    print(f'################# [ K-Fold {i+1} ] #################')
                    
                    # Early Stopping
                    early_stopping = PyTorch().EarlyStopping(
                        patience = self.early_stopping_p, 
                        mode = self.early_stopping_mode, 
                        save_path = self.save_path_model
                    )
                    
                    # Metrics for epochs
                    avg_loss_t, avg_accuracy_t, avg_recall_t, avg_auc_t, avg_precision_t, avg_npv_t = [], [], [], [], [], []
                    avg_loss_v, avg_accuracy_v, avg_recall_v, avg_auc_v, avg_precision_v, avg_npv_v = [], [], [], [], [], []
                    # Epochs
                    for epoch in range(0, self.max_epochs):

                    # Checking the learning rate according to the epochs:
                    #print(f'\nEpoch {epoch + 1} -- Kfold: {i + 1}\n--------------------------------------------------------------')
                    #print(f"Epoch {epoch + 1} -- Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

                        # Metrics Training
                        train_loss = 0.0 
                        train_steps = 0 
                        accuracy_train.reset()
                        recall_train.reset()
                        auc_train.reset()
                        precision_train.reset()
                        npv_train.reset()

                        # Metrics Validation
                        val_loss = 0.0
                        val_steps = 0
                        accuracy_val.reset()
                        recall_val.reset()
                        auc_val.reset()
                        precision_val.reset()
                        npv_val.reset()

                        # Target Score
                        target_score.reset()

                        # Save preds and labels
                        preds_val, labels_val = [], []
                        
                        # Training
                        net.train()

                        for cat_input, num_input, labels in (trainloader):
                
                            # Inputs + Labels to(device)    
                            cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                            
                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Foward Pass
                            outputs = net(cat_input, num_input)
                            loss = criterion(outputs, labels)

                            # Backward + optimize
                            loss.backward()
                            # Optimizer
                            optimizer.step()

                            # Accumulating Loss
                            train_loss += loss.item()
                            train_steps += 1

                            # Updating metrics
                            accuracy_train.update(torch.sigmoid(outputs), labels.int())
                            recall_train.update(torch.sigmoid(outputs), labels.int())
                            auc_train.update(torch.sigmoid(outputs), labels.int())
                            precision_train.update(torch.sigmoid(outputs), labels.int())
                            npv_train.update(torch.sigmoid(outputs), labels.int())

                        # Evaluation
                        net.eval()

                        # Disabling gradient calculations
                        with torch.no_grad():
                            # Get the inputs; data is a list of [inputs, labels]
                            for cat_input, num_input, labels in (valloader):
                                    
                                    # Inputs + Labels to(device)
                                    cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                    
                                    # Eval net
                                    outputs = net(cat_input, num_input)
                                    loss = criterion(outputs, labels)
                                    
                                    # Accumulating Loss
                                    val_loss += loss.item()
                                    val_steps += 1

                                    # Updating metrics
                                    accuracy_val.update(torch.sigmoid(outputs), labels.int())
                                    recall_val.update(torch.sigmoid(outputs), labels.int())
                                    auc_val.update(torch.sigmoid(outputs), labels.int())
                                    precision_val.update(torch.sigmoid(outputs), labels.int())
                                    npv_val.update(torch.sigmoid(outputs), labels.int())
                                    
                                    # Target Score
                                    target_score.update(torch.sigmoid(outputs), labels.int())

                        # Saving metrics by epoch
                        # Train
                        avg_accuracy_t.append(accuracy_train.compute().item())
                        avg_recall_t.append(recall_train.compute().item())
                        avg_auc_t.append(auc_train.compute().item())
                        avg_precision_t.append(precision_train.compute().item())
                        avg_npv_t.append(npv_train.compute().item())
                        avg_loss_t.append(train_loss / train_steps)
                        # Validation
                        avg_accuracy_v.append(accuracy_val.compute().item())
                        avg_recall_v.append(recall_val.compute().item())
                        avg_auc_v.append(auc_val.compute().item())
                        avg_precision_v.append(precision_val.compute().item())
                        avg_npv_v.append(npv_val.compute().item())
                        avg_loss_v.append(val_loss / val_steps)
                                    
                        # Scheduler step
                        scheduler.step()

                        # Early_stopping step
                        early_stopping(score = target_score.compute().item(), model = net, epoch = epoch)
                        # Stopping 
                        if early_stopping.early_stop:
                            print(f'>>>>>>> Finished Training K-Fold {i+1}.')
                        
                            # Final Validation Score Model
                            # Load the Best Model
                            net.load_state_dict(torch.load(self.save_path_model))
                            

                            # Metrics Validation
                            val_loss = 0.0
                            val_steps = 0
                            accuracy_val.reset()
                            recall_val.reset()
                            auc_val.reset()
                            precision_val.reset()
                            npv_val.reset()
                            
                            # Evaluation
                            net.eval()
                            # Disabling gradient calculations
                            with torch.no_grad():
                                # Get the inputs; data is a list of [inputs, labels]
                                for cat_input, num_input, labels in (valloader):
                                        
                                        # Inputs + Labels to(device)
                                        cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                        
                                        # Eval net
                                        outputs = net(cat_input, num_input)
                                        loss = criterion(outputs, labels)
                                        
                                        # Accumulating Loss
                                        val_loss += loss.item()
                                        val_steps += 1

                                        # Updating metrics
                                        accuracy_val.update(torch.sigmoid(outputs), labels.int())
                                        recall_val.update(torch.sigmoid(outputs), labels.int())
                                        auc_val.update(torch.sigmoid(outputs), labels.int())
                                        precision_val.update(torch.sigmoid(outputs), labels.int())
                                        npv_val.update(torch.sigmoid(outputs), labels.int())
                                        
                                        # Accumulating Predictions
                                        preds_val.append(torch.sigmoid(outputs).detach().cpu())
                                        labels_val.append(labels.detach().cpu())
                        
                            # Concatenates all batches
                            preds_val = torch.cat(preds_val).float()
                            labels_val = torch.cat(labels_val).long()

                            break
                    
                    # Metrics out
                    print('\nTrain Metrics:') 
                    print(f'Loss: {train_loss / train_steps:.3f}')   
                    print(f'Accuracy: {accuracy_train.compute().item() * 100:> 0.1f}%') 
                    print(f'Precision: {precision_train.compute().item() * 100:> 0.1f}%')
                    print(f'NPV: {npv_train.compute().item() * 100:> 0.1f}%')
                    print(f'Recall: {recall_train.compute().item() *100:> 0.1f}%') 
                    print(f'AUC-ROC: {auc_train.compute().item() *100:> 0.1f}%') 

                    print('\nValidation Metrics:')
                    print(f'Loss: {val_loss / val_steps:.3f}')
                    print(f'Accuracy: {accuracy_val.compute().item() * 100:> 0.1f}%')
                    print(f'Precision: {precision_val.compute().item() * 100:> 0.1f}%')
                    print(f'NPV: {npv_val.compute().item() * 100:> 0.1f}%')
                    print(f'Recall: {recall_val.compute().item() * 100:> 0.1f}%')
                    print(f'AUC-ROC: {auc_val.compute().item() * 100:> 0.1f}%')
                    
                    # Graphics for Training and Validation
            
                    self._plot_metrics(
                        avg_loss_t = avg_loss_t, 
                        avg_loss_v = avg_loss_v,
                        avg_accuracy_t = avg_accuracy_t, 
                        avg_accuracy_v = avg_accuracy_v,
                        avg_precision_t = avg_precision_t, 
                        avg_precision_v = avg_precision_v,
                        avg_npv_t = avg_npv_t, 
                        avg_npv_v = avg_npv_v,
                        avg_recall_t = avg_recall_t, 
                        avg_recall_v = avg_recall_v,
                        avg_auc_t = avg_auc_t, 
                        avg_auc_v = avg_auc_v
                    )

                    self._confusion_matrix(
                        preds = preds_val,
                        labels = labels_val,
                        title = f'Confusion Matrix is K-Fold: {i + 1}'
                    )

                    # Time sleep
                    time.sleep(5)

                    # Calculating metrics per fold
                    accuracy_kfolds.append(accuracy_val.compute().item())
                    recall_kfolds.append(recall_val.compute().item())
                    auc_kfolds.append(auc_val.compute().item())
                    precision_kfolds.append(precision_val.compute().item())
                    npv_kfolds.append(npv_val.compute().item())
                    loss_kfolds.append(val_loss / val_steps)

                print('\n\n✅### Cross validation Metrics ### :')

                print(f'\n🔴 Loss: {np.mean(loss_kfolds):.3f}') 
                print(f'☑️ Standard Deviation - Loss: {np.std(loss_kfolds):.6f}')

                print(f'\n🟠 Accuracy: {(np.mean(accuracy_kfolds) * 100):.2f}%')
                print(f'☑️ Standard Deviation - Accuracy: {np.std(accuracy_kfolds):.6f}')

                print(f'\n🔵 Precision: {(np.mean(precision_kfolds) * 100):.2f}%')
                print(f'☑️ Standard Deviation - Precision: {np.std(precision_kfolds):.6f}')

                print(f'\n🔵 NPV: {(np.mean(npv_kfolds) * 100):.2f}%')
                print(f'☑️ Standard Deviation - NPV: {np.std(npv_kfolds):.6f}')

                print(f'\n⚠️ Recall: {(np.mean(recall_kfolds) * 100):.2f}%')
                print(f'☑️ Standard Deviation - Recall: {np.std(recall_kfolds):.6f}')

                print(f'\n🎯 AUC-ROC: {(np.mean(auc_kfolds) * 100):.2f}%')
                print(f'☑️ Standard Deviation - AUC-ROC: {np.std(auc_kfolds):.6f}')

            except Exception as e:
                print(f'[ERROR] Cross validation flow execution failed: {str(e)}.')

        # ======================================================== #
        # Flow Hyper Tunning                                       #
        # ======================================================== #

        # ======================================================== #
        # Cross Tunning - Function                                 #
        # ======================================================== #
        def _cross_tunning(
            self,
            l1: int,
            l2: int,
            l3: int,
            batch_size: int,
            lr: float,
            weight_decay: float,
        
        ):
            """
            Perform k-fold cross-validation with hyperparameter configuration for a PyTorch model.

            This function evaluates a neural network using k-fold cross-validation with
            oversampling, learning rate scheduling, and early stopping. It is typically called
            within a hyperparameter search routine (e.g., Optuna or Ray Tune).

            Args:
                l1 (int): Number of units in the first hidden layer.
                l2 (int): Number of units in the second hidden layer.
                l3 (int): Number of units in the third hidden layer.
                batch_size (int): Batch size for DataLoader during training and validation.
                lr (float): Learning rate for the optimizer.
                weight_decay (float): L2 regularization coefficient for the optimizer.

            Returns:
                float: Mean AUC-ROC score across all folds.

            Attributes Used:
                self.trainset: Training dataset used for k-fold splitting.
                self.k_fold (int): Number of folds for cross-validation.
                self.num_workers (int): Number of workers for DataLoader.
                self.dropout_rate (float): Dropout rate applied to the network.
                self.alpha (float): Alpha parameter for Focal Loss.
                self.gamma (float): Gamma parameter for Focal Loss.
                self.max_epochs (int): Maximum number of training epochs.
                self.early_stopping_p (int): Patience parameter for early stopping.
                self.early_stopping_mode (str): Mode for early stopping ('min' or 'max').
                self.target_score (str): Performance metric for early stopping 
                    ('accuracy', 'recall', 'roc', 'precision', 'npv').

            Metrics Computed:
                - Accuracy
                - Recall
                - Precision
                - AUC-ROC
                - Negative Predictive Value (NPV)
                - Loss

            Raises:
                Exception: If any error occurs during training or evaluation.
            """
            try:

                folds = self._kfolds()
                # Defining list to store fold metrics
                auc_kfolds = []

                # Initialize Metrics for binary classification

                # Validation
                auc_val = BinaryAUROC(thresholds = None)

                # Score target
                if self.target_score == 'accuracy':
                    target_score = BinaryAccuracy()
                
                elif self.target_score == 'recall':
                    target_score = BinaryRecall()
                
                elif self.target_score == 'roc':
                    target_score = BinaryAUROC(thresholds = None)
                
                elif self.target_score == 'precision':
                    target_score = BinaryPrecision()

                elif self.target_score == 'npv':
                    target_score = BinaryNegativePredictiveValue()

                for i in range(self.k_fold):

                    # Separating training and validation data
                    # Fold for validation
                    val_set = folds[i] 

                    # All training folds except validation fold
                    train_sets = [folds[j] for j in range(self.k_fold) if j != i]
                    train_set = torch.utils.data.ConcatDataset(train_sets)

                    sampler, prop_class_positive =  self._oversampling(trainset = train_set)
                    
                    # Train Loader
                    trainloader = torch.utils.data.DataLoader(
                        train_set, 
                        batch_size = batch_size, 
                        sampler = sampler, 
                        num_workers = self.num_workers,
                        drop_last = True,
                    )

                    # Val Loader
                    valloader = torch.utils.data.DataLoader(
                        val_set, 
                        batch_size = batch_size, 
                        shuffle = False, 
                        num_workers = self.num_workers, 
                        drop_last = False,
                    )
                    

            
                    # Loading Net
                    net = PyTorch.Net(
                        l1 = l1,
                        l2 = l2, 
                        l3 = l3, 
                        dropout_rate = self.dropout_rate,
                        prior_minoritary_class = prop_class_positive,  
                    )
                    # Moving the network to the device
                    net, device = self._device(net)

                    # Criterion
                    criterion = PyTorch().FocalLoss(alpha = self.alpha, gamma = self.gamma).to(device)

                    # Optimizer
                    optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay = weight_decay) 

                    # Scheduler
                    # Warmup (linear from 1e-5 to 0.001)
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor = 0.01,
                        total_iters = int(self.max_epochs * 0.05),
                    )
                    # Cosine Annealing after warmup
                    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max = self.max_epochs - int(self.max_epochs * 0.05),
                        eta_min = 1e-6,
                    )
                    # Composition: 10 warmup epochs + (max_epochs cosine - 10 warmup)
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers = [warmup_scheduler, cosine_scheduler],
                        milestones = [int(self.max_epochs * 0.05)],
                    )
                    # Adjusting error caused by scheduler
                    warnings.filterwarnings('ignore', category = UserWarning)

                    # Saving the model temporarily with early stopping
                    with tempfile.NamedTemporaryFile(delete = False) as temp_model_file:  
                        
                        # Early Stopping
                        early_stopping = PyTorch().EarlyStopping(
                            patience = self.early_stopping_p, 
                            mode = self.early_stopping_mode, 
                            save_path = temp_model_file,
                            tempfile_save = True,
                            verbose = False
                        )
                        
                        # Epochs
                        for epoch in range(0, self.max_epochs):

                            # Target Score
                            target_score.reset()
                            
                            # Training
                            net.train()

                            for cat_input, num_input, labels in (trainloader):
                    
                                # Inputs + Labels to(device)    
                                cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                
                                # Zero the parameter gradients
                                optimizer.zero_grad()

                                # Foward Pass
                                outputs = net(cat_input, num_input)
                                loss = criterion(outputs, labels)

                                # Backward + optimize
                                loss.backward()
                                # Optimizer
                                optimizer.step()

                            # Evaluation
                            net.eval()

                            # Disabling gradient calculations
                            with torch.no_grad():
                                # Get the inputs; data is a list of [inputs, labels]
                                for cat_input, num_input, labels in (valloader):
                                        
                                        # Inputs + Labels to(device)
                                        cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                        
                                        # Eval net
                                        outputs = net(cat_input, num_input)
                                        
                                        # Target Score
                                        target_score.update(torch.sigmoid(outputs), labels.int())

                            # Scheduler step
                            scheduler.step()

                            # Early_stopping step
                            early_stopping(score = target_score.compute().item(), model = net, epoch = epoch)
                            # Stopping 
                            if early_stopping.early_stop:
                    
                                # Final Validation Score Model
                                # Load the Best Model
                                net.load_state_dict(torch.load(temp_model_file.name))
                                os.remove(temp_model_file.name)

                                # AUC-ROC
                                auc_val.reset()
                              
                                # Evaluation
                                net.eval()
                                # Disabling gradient calculations
                                with torch.no_grad():
                                    # Get the inputs; data is a list of [inputs, labels]
                                    for cat_input, num_input, labels in (valloader):
                                            
                                            # Inputs + Labels to(device)
                                            cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                            
                                            # Eval net
                                            outputs = net(cat_input, num_input)

                                            auc_val.update(torch.sigmoid(outputs), labels.int())

                                break
                            
                    # Calculating AUC-ROC per fold
                    auc_kfolds.append(auc_val.compute().item())
                        
                return np.mean(auc_kfolds)

                print('✅Finished Training')

            except Exception as e:
                print(f'[ERROR] Failed to execute cross adjustment flow: {str(e)}.')

        # ======================================================== #
        # HyperTunning - Function                                  #
        # ======================================================== #
        def HyperTunning(
            self,
            n_samples: int = 10,
        ):
            """
            Perform hyperparameter tuning using Optuna.

            This method defines a search space for model architecture and training
            hyperparameters, then uses Optuna to optimize them via k-fold cross-validation.
            The optimization target is the mean AUC-ROC score.

            Args:
                n_samples (int, optional): Number of trials (hyperparameter configurations)
                    to evaluate. Default is 10.

            Returns:
                dict: Best hyperparameter configuration found. Includes:
                    - 'l1' (int): Units in the first hidden layer.
                    - 'l2' (int): Units in the second hidden layer.
                    - 'l3' (int): Units in the third hidden layer.
                    - 'batch_size' (int): Training batch size.
                    - 'lr' (float): Learning rate.
                    - 'weight_decay' (float): Weight decay (L2 regularization).

            Workflow:
                1. Define hyperparameter search space for layer sizes, batch size, learning
                    rate, and weight decay.
                2. Run Optuna optimization with Tree-structured Parzen Estimator (TPE).
                3. Evaluate each trial using `_cross_tunning`.
                4. Select the configuration with the best mean AUC-ROC.

            Raises:
                Exception: Propagates errors if tuning fails inside `_cross_tunning`.
            """
            try:
            
                def objective(
                    trial
                ):
                    # Search space
                    l1 = trial.suggest_int('l1', 32, 512, step = 32)
                    l2 = trial.suggest_int('l2', 16, 512, step = 16)
                    l3 = trial.suggest_int('l3', 8, 256, step = 8)
                    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
                    lr = trial.suggest_float('lr', 1e-3, 1e-2, log = True)
                    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 5e-4]) 

                    # Assessment via cross-validation
                    auc_mean = self._cross_tunning(
                        l1 = l1,
                        l2 = l2,
                        l3 = l3,
                        batch_size = batch_size,
                        lr = lr,
                        weight_decay = weight_decay,
                    
                    )

                    return auc_mean
                
                # Optuna Study
                study = optuna.create_study(
                    direction = 'maximize',
                    sampler = optuna.samplers.TPESampler(seed = 10),
                )

                study.optimize(objective, n_trials = n_samples)

                print('\n✅ Best trial:')
                print(f'Score: {study.best_value}')
                print(f'Params:{study.best_params}')

                return study.best_params
            
            except Exception as e:
                print(f'[ERROR] Failed to execute hyperparameter tunning: {str(e)}.')

        # ======================================================== #
        # Final Training - Function                                #
        # ======================================================== #
        def FinalTraining(
            self,
        ):
            """
            Executes the final training routine for the neural network, including 
            data preparation, training, validation, early stopping, and metric 
            reporting.

            This method:
            - Splits the dataset into training and validation sets.
            - Performs oversampling to handle class imbalance.
            - Initializes the model, loss function, optimizer, and learning rate schedulers.
            - Trains the model with early stopping based on a selected target score.
            - Evaluates the model on a validation set.
            - Plots metrics and the final confusion matrix.

            The training loop collects accuracy, precision, recall, NPV, AUC, and loss
            for both training and validation sets across epochs. Learning rate scheduling 
            uses a warm-up phase followed by cosine annealing.

            Args:
                self: 
                    An instance of the training class containing:
                        - trainset (Dataset): The dataset used for training and validation split.
                        - target_score (str): The metric used for early stopping ('accuracy', 'recall', 'roc', 'precision', or 'npv').
                        - batch_size (int): Batch size for training and validation data loaders.
                        - num_workers (int): Number of worker threads for data loading.
                        - seed (int): Random seed for reproducibility in splitting.
                        - l1, l2, l3 (int): Sizes of the hidden layers in the network.
                        - dropout_rate (float): Dropout probability for regularization.
                        - lr (float): Initial learning rate for the optimizer.
                        - weight_decay (float): Weight decay (L2 regularization) for the optimizer.
                        - max_epochs (int): Maximum number of training epochs.
                        - early_stopping_p (int): Patience for early stopping.
                        - early_stopping_mode (str): Mode for early stopping ('min' or 'max').
                        - save_path_model (str): File path to save the best model weights.

            Raises:
                Exception: If any part of the training process fails, an error message 
                will be printed containing the exception details.

            Prints:
                - Class imbalance ratio in the training set.
                - Final metrics for training and validation sets (Loss, Accuracy, Precision, 
                NPV, Recall, AUC-ROC).

            Side Effects:
                - Saves the best-performing model to disk.
                - Generates and displays plots for metrics and the final confusion matrix.
            """
            try:

                # Initialize Metrics for binary classification

                # Training
                accuracy_train = BinaryAccuracy()
                recall_train = BinaryRecall()
                auc_train = BinaryAUROC(thresholds = None)
                precision_train = BinaryPrecision()
                npv_train = BinaryNegativePredictiveValue()

                # Validation
                accuracy_val = BinaryAccuracy()
                recall_val = BinaryRecall()
                auc_val = BinaryAUROC(thresholds = None)
                precision_val = BinaryPrecision()
                npv_val = BinaryNegativePredictiveValue()

                # Score target
                if self.target_score == 'accuracy':
                    target_score = BinaryAccuracy()
                
                elif self.target_score == 'recall':
                    target_score = BinaryRecall()
                
                elif self.target_score == 'roc':
                    target_score = BinaryAUROC(thresholds = None)
                
                elif self.target_score == 'precision':
                    target_score = BinaryPrecision()

                elif self.target_score == 'npv':
                    target_score = BinaryNegativePredictiveValue()

                

                # Separating training and validation data
                    # Spliting trainset and validationset

                train_size = int(len(self.trainset) * 0.8)
                train_set, val_set = torch.utils.data.random_split(
                    self.trainset, [train_size, len(self.trainset) - train_size],
                    generator = torch.Generator().manual_seed(self.seed)
                )

                sampler, prop_class_positive =  self._oversampling(trainset = train_set)
                
                # Train Loader
                trainloader = torch.utils.data.DataLoader(
                    train_set, 
                    batch_size = self.batch_size, 
                    sampler = sampler, 
                    num_workers = self.num_workers,
                    drop_last = True,
                )

                # Val Loader
                valloader = torch.utils.data.DataLoader(
                    val_set, 
                    batch_size = self.batch_size, 
                    shuffle = False, 
                    num_workers = self.num_workers, 
                    drop_last = False,
                )
                

        
                # Loading Net
                net = PyTorch.Net(
                    l1 = self.l1,
                    l2 = self.l2, 
                    l3 = self.l3, 
                    dropout_rate = self.dropout_rate,
                    prior_minoritary_class = prop_class_positive,  
                )
                # Moving the network to the device
                net, device = self._device(net)

                # Criterion
                criterion = PyTorch().FocalLoss(alpha = self.alpha, gamma = self.gamma).to(device)

                # Optimizer
                optimizer = optim.AdamW(net.parameters(), lr = self.lr, weight_decay = self.weight_decay) 

                # Scheduler
                # Warmup (linear from 1e-5 to 0.001)
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor = 0.01,
                    total_iters = int(self.max_epochs * 0.05),
                )
                # Cosine Annealing after warmup
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max = self.max_epochs - int(self.max_epochs * 0.05),
                    eta_min = 1e-6,
                )
                # Composition: 10 warmup epochs + (max_epochs cosine - 10 warmup)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers = [warmup_scheduler, cosine_scheduler],
                    milestones = [int(self.max_epochs * 0.05)],
                )

                # Adjusting error caused by scheduler
                warnings.filterwarnings('ignore', category = UserWarning)
                
                # Early Stopping
                early_stopping = PyTorch().EarlyStopping(
                    patience = self.early_stopping_p, 
                    mode = self.early_stopping_mode, 
                    save_path = self.save_path_model
                )
                
                # Metrics for epochs
                avg_loss_t, avg_accuracy_t, avg_recall_t, avg_auc_t, avg_precision_t, avg_npv_t = [], [], [], [], [], []
                avg_loss_v, avg_accuracy_v, avg_recall_v, avg_auc_v, avg_precision_v, avg_npv_v = [], [], [], [], [], []
                
                # Epochs
                for epoch in range(0, self.max_epochs):

                    # Metrics Training
                    train_loss = 0.0 
                    train_steps = 0 
                    accuracy_train.reset()
                    recall_train.reset()
                    auc_train.reset()
                    precision_train.reset()
                    npv_train.reset()

                    # Metrics Validation
                    val_loss = 0.0
                    val_steps = 0
                    accuracy_val.reset()
                    recall_val.reset()
                    auc_val.reset()
                    precision_val.reset()
                    npv_val.reset()

                    # Target Score
                    target_score.reset()

                    # Save preds and labels
                    preds_val, labels_val = [], []
                    
                    # Training
                    net.train()

                    for cat_input, num_input, labels in (trainloader):
            
                        # Inputs + Labels to(device)    
                        cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                        
                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Foward Pass
                        outputs = net(cat_input, num_input)
                        loss = criterion(outputs, labels)

                        # Backward + optimize
                        loss.backward()
                        # Optimizer
                        optimizer.step()

                        # Accumulating Loss
                        train_loss += loss.item()
                        train_steps += 1

                        # Updating metrics
                        accuracy_train.update(torch.sigmoid(outputs), labels.int())
                        recall_train.update(torch.sigmoid(outputs), labels.int())
                        auc_train.update(torch.sigmoid(outputs), labels.int())
                        precision_train.update(torch.sigmoid(outputs), labels.int())
                        npv_train.update(torch.sigmoid(outputs), labels.int())

                    # Evaluation
                    net.eval()

                    # Disabling gradient calculations
                    with torch.no_grad():
                        # Get the inputs; data is a list of [inputs, labels]
                        for cat_input, num_input, labels in (valloader):
                                
                                # Inputs + Labels to(device)
                                cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                
                                # Eval net
                                outputs = net(cat_input, num_input)
                                loss = criterion(outputs, labels)
                                
                                # Accumulating Loss
                                val_loss += loss.item()
                                val_steps += 1

                                # Updating metrics
                                accuracy_val.update(torch.sigmoid(outputs), labels.int())
                                recall_val.update(torch.sigmoid(outputs), labels.int())
                                auc_val.update(torch.sigmoid(outputs), labels.int())
                                precision_val.update(torch.sigmoid(outputs), labels.int())
                                npv_val.update(torch.sigmoid(outputs), labels.int())
                                
                                # Target Score
                                target_score.update(torch.sigmoid(outputs), labels.int())

                    # Saving metrics by epoch
                    # Train
                    avg_accuracy_t.append(accuracy_train.compute().item())
                    avg_recall_t.append(recall_train.compute().item())
                    avg_auc_t.append(auc_train.compute().item())
                    avg_precision_t.append(precision_train.compute().item())
                    avg_npv_t.append(npv_train.compute().item())
                    avg_loss_t.append(train_loss / train_steps)
                    # Validation
                    avg_accuracy_v.append(accuracy_val.compute().item())
                    avg_recall_v.append(recall_val.compute().item())
                    avg_auc_v.append(auc_val.compute().item())
                    avg_precision_v.append(precision_val.compute().item())
                    avg_npv_v.append(npv_val.compute().item())
                    avg_loss_v.append(val_loss / val_steps)
                                
                    # Scheduler step
                    scheduler.step()

                    # Early_stopping step
                    early_stopping(score = target_score.compute().item(), model = net, epoch = epoch)
                    # Stopping 
                    if early_stopping.early_stop:
                        print(f'>>>>>>> Finished Training.')
                    
                        # Final Validation Score Model
                        # Load the Best Model
                        net.load_state_dict(torch.load(self.save_path_model))
                        

                        # Metrics Validation
                        val_loss = 0.0
                        val_steps = 0
                        accuracy_val.reset()
                        recall_val.reset()
                        auc_val.reset()
                        precision_val.reset()
                        npv_val.reset()
                        
                        # Evaluation
                        net.eval()
                        # Disabling gradient calculations
                        with torch.no_grad():
                            # Get the inputs; data is a list of [inputs, labels]
                            for cat_input, num_input, labels in (valloader):
                                    
                                    # Inputs + Labels to(device)
                                    cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                                    
                                    # Eval net
                                    outputs = net(cat_input, num_input)
                                    loss = criterion(outputs, labels)
                                    
                                    # Accumulating Loss
                                    val_loss += loss.item()
                                    val_steps += 1

                                    # Updating metrics
                                    accuracy_val.update(torch.sigmoid(outputs), labels.int())
                                    recall_val.update(torch.sigmoid(outputs), labels.int())
                                    auc_val.update(torch.sigmoid(outputs), labels.int())
                                    precision_val.update(torch.sigmoid(outputs), labels.int())
                                    npv_val.update(torch.sigmoid(outputs), labels.int())
                                    
                                    # Accumulating Predictions
                                    preds_val.append(torch.sigmoid(outputs).detach().cpu())
                                    labels_val.append(labels.detach().cpu())
                    
                        # Concatenates all batches
                        preds_val = torch.cat(preds_val).float()
                        labels_val = torch.cat(labels_val).long()

                        break

                # Distribution of the minority class
                print(f'\n⚖️ The Distribution of the minority class (prop_class_positive): {prop_class_positive}')

                # Metrics out
                print('\n✅ Train Metrics:') 
                print(f'Loss: {train_loss / train_steps:.3f}')   
                print(f'Accuracy: {accuracy_train.compute().item() * 100:> 0.1f}%') 
                print(f'Precision: {precision_train.compute().item() * 100:> 0.1f}%')
                print(f'NPV: {npv_train.compute().item() * 100:> 0.1f}%')
                print(f'Recall: {recall_train.compute().item() *100:> 0.1f}%') 
                print(f'AUC-ROC: {auc_train.compute().item() *100:> 0.1f}%') 

                print('\n☑️ Validation Metrics:')
                print(f'Loss: {val_loss / val_steps:.3f}')
                print(f'Accuracy: {accuracy_val.compute().item() * 100:> 0.1f}%')
                print(f'Precision: {precision_val.compute().item() * 100:> 0.1f}%')
                print(f'NPV: {npv_val.compute().item() * 100:> 0.1f}%')
                print(f'Recall: {recall_val.compute().item() * 100:> 0.1f}%')
                print(f'AUC-ROC: {auc_val.compute().item() * 100:> 0.1f}%')
                
                # Graphics for Training and Validation
                self._plot_metrics(
                    avg_loss_t = avg_loss_t, 
                    avg_loss_v = avg_loss_v,
                    avg_accuracy_t = avg_accuracy_t, 
                    avg_accuracy_v = avg_accuracy_v,
                    avg_precision_t = avg_precision_t, 
                    avg_precision_v = avg_precision_v,
                    avg_npv_t = avg_npv_t, 
                    avg_npv_v = avg_npv_v,
                    avg_recall_t = avg_recall_t, 
                    avg_recall_v = avg_recall_v,
                    avg_auc_t = avg_auc_t, 
                    avg_auc_v = avg_auc_v
                )

                self._confusion_matrix(
                    preds = preds_val,
                    labels = labels_val,
                    title = 'Confusion Matrix Final Validation'
                )

            except Exception as e:
                print(f'[ERROR] Failed to execute final training flow: {str(e)}.')

        # ======================================================== #
        # Final Test     - Function                                #
        # ======================================================== #
        def FinalTest(
            self,
            net,
        ):
            """
            Executes the final evaluation of a trained neural network model on the test dataset,
            computing multiple performance metrics and displaying a confusion matrix.

            This method:
                - Loads the trained model onto the appropriate device (CPU or GPU).
                - Iterates over the test dataset without gradient computation.
                - Calculates the loss and multiple binary classification metrics.
                - Stores predictions and ground truth labels for later analysis.
                - Displays the final evaluation results and a confusion matrix.

            Args:
                net (torch.nn.Module):
                    The trained PyTorch neural network model to be evaluated.

            Returns:
                tuple:
                    - preds_test (torch.Tensor): Tensor containing all predicted probabilities for the test dataset.
                    - labels_test (torch.Tensor): Tensor containing all ground truth labels for the test dataset.

            Raises:
                Exception:
                    If any error occurs during the evaluation process.

            Notes:
                - This method uses a `FocalLoss` criterion for evaluation.
                - Metrics computed:
                    * Binary Accuracy
                    * Binary Precision
                    * Binary Negative Predictive Value (NPV)
                    * Binary Recall
                    * Binary Area Under the ROC Curve (AUC-ROC)
                - Predictions are probability scores obtained from `torch.sigmoid(outputs)`.
                - The confusion matrix is generated via the `_confusion_matrix` method.
            """
            try:
                
                # Net Loading
                net, device = self._device(net)

                # Test Loader
                testloader = torch.utils.data.DataLoader(
                    self.testset, batch_size = 4, shuffle = False, num_workers = 2, 
                    drop_last = False,
                )
                
                # Criterion
                criterion = PyTorch().FocalLoss(alpha = self.alpha, gamma = self.gamma).to(device)

                # Evaluation
                net.eval()

                # Metrics Testing
                test_loss = 0.0 
                test_steps = 0 
                
                accuracy_test = BinaryAccuracy()
                precision_test = BinaryPrecision()
                npv_test = BinaryNegativePredictiveValue()
                recall_test = BinaryRecall()
                auc_test = BinaryAUROC(thresholds = None)
                
                # Save preds and labels
                preds_test, labels_test = [], []

                # Disabling gradient calculations
                with torch.no_grad():
                    # Get the inputs; data is a list of [inputs, labels]
                    for cat_input, num_input, labels in (testloader):
                            
                            # Inputs + Labels to(device)
                            cat_input, num_input, labels = cat_input.to(device), num_input.to(device), labels.to(device)
                            
                            # Eval net
                            outputs = net(cat_input, num_input)
                            loss = criterion(outputs, labels)
                            
                            # Updating Metrics
                            accuracy_test.update(torch.sigmoid(outputs), labels.int())
                            precision_test.update(torch.sigmoid(outputs), labels.int())
                            npv_test.update(torch.sigmoid(outputs), labels.int())
                            recall_test.update(torch.sigmoid(outputs), labels.int())
                            auc_test.update(torch.sigmoid(outputs), labels.int())

                            # Accumulating Loss
                            test_loss += loss.item()
                            test_steps += 1
                            # Accumulating Predictions
                            preds_test.append(torch.sigmoid(outputs).detach().cpu())
                            labels_test.append(labels.detach().cpu())

                # Concatenates all batches
                preds_test = torch.cat(preds_test).float()
                labels_test = torch.cat(labels_test).long()
                
                # Metrics Out
                print('\n✅ Test Metrics:')
                print(f'Loss: {test_loss / test_steps:.3f}')
                print(f'Accuracy: {accuracy_test.compute().item() * 100:> 0.1f}%')
                print(f'Precision: {precision_test.compute().item() * 100:> 0.1f}%')
                print(f'NPV: {npv_test.compute().item() * 100:> 0.1f}%')
                print(f'Recall: {recall_test.compute().item() * 100:> 0.1f}%')
                print(f'AUC-ROC: {auc_test.compute().item() * 100:> 0.1f}%')

                self._confusion_matrix(
                    preds = preds_test,
                    labels = labels_test,
                    title = 'Confusion Matrix Final Test'
                )
                # Return Preds
                return preds_test, labels_test

            except Exception as e:
                print(f'[ERROR] Failed to execute test flow: {str(e)}.')