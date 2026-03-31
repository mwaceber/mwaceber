import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
from sklearn.linear_model import (LinearRegression, LogisticRegression, Lasso)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from sklearn.model_selection import (train_test_split, GridSearchCV)
import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset
from torchmetrics import (MeanAbsoluteError, R2Score)
from torchinfo import summary
from torchvision.io import read_image
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR100
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import seed_everything
seed_everything(0, workers = True)
torch.use_deterministic_algorithms(True, warn_only=True)
from torchvision.transforms import (Resize, Normalize, CenterCrop,ToTensor)
from ISLP.torch import (SimpleDataModule, SimpleModule, ErrorTracker, rec_num_workers)
from ISLP.torch.imdb import (load_lookup, load_tensor, load_sparse, load_sequential)
from glob import glob
import json


#Single Layer Network on Hitters Data
Hitters = load_data('Hitters').dropna()
n = Hitters.shape[0] # We use mean absolute error on validation dataset to compare performance
# of least squares and the lasso to that of neural network
model = MS(Hitters.columns.drop('Salary'), intercept= False)
X = model.fit_transform(Hitters).to_numpy()
Y = Hitters['Salary'].to_numpy() #to_numpy() converts pandas dataframe to numpy arrays since we'll
# use sklearn to fit the lasso model
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size= (1/3), random_state= 1)
#Evaluate the test error after fittin the linear model
hit_lm = LinearRegression().fit(X_train, Y_train)
Yhat_test = hit_lm.predict(X_test)
MAE = np.abs(Yhat_test - Y_test).mean()
#Encode a pipeline by normalizing the features(usin stdscaler()) then fit the lasso without normalizing
scaler = StandardScaler(with_mean=True, with_std=True)
lasso = Lasso(warm_start=True, max_iter=30000)
standard_lasso = Pipeline(steps= [('scaler', scaler), ('lasso', lasso)])
#create a grid of values for lambda(lam_max)
X_s = scaler.fit_transform(X_train)
n = X_s.shape[0]
lam_max = np.fabs(X_s.T.dot(Y_train - Y_train.mean())).max() / n
param_grid = {'alpha': np.exp(np.linspace(0, np.log(0.01), 100)) * lam_max}
#Perform cross-validation using sequence of lambda values
cv = KFold(10, shuffle=True, random_state=1)
grid = GridSearchCV(lasso, param_grid, cv=cv, scoring= 'neg_mean_absolute_error')
grid.fit(X_train, Y_train)
#Extract the lasso model with best cross-validated mean absolute error and evaluate performance 
# on X_test and Y_test
trained_lasso = grid.best_estimator_
Yhat_test = trained_lasso.predict(X_test)
np.fabs(Yhat_test - Y_test).mean()
#Specifying a network: Classes and inheritance
class HittersModel(nn.Module):
    def __init__(self, input_size):
        super(HittersModel, self).__init__()
        self.flatten = nn.Flatten()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(50, 1))
    
    def forward(self, x):
        x = self.flatten(x)
        return torch.flatten(self.sequential(x))
hit_model = HittersModel(X.shape[1])
summary(hit_model, input_size=X_train.shape,col_names = ['input_size', 'output_size', 'num_params'])
# Transform our training data and test data in form accessible to torch
X_train_t = torch.tensor(X_train.astype(np.float32))
Y_train_t = torch.tensor(Y_train.astype(np.float32))
hit_train = TensorDataset(X_train_t, Y_train_t)
X_test_t = torch.tensor(X_test.astype(np.float32))
Y_test_t = torch.tensor(Y_test.astype(np.float32))
hit_test = TensorDataset(X_test_t, Y_test_t)
max_num_workers = rec_num_workers()
# Now we pass the dataset into the DataLoader() which ultimately passes data into our network
hit_dm = SimpleDataModule(hit_train, hit_test, batch_size= 32, num_workers=min(4, max_num_workers),
                            validation= hit_test)
#hit_module specifies the netowrk architecyure as well as train/validating/test steps
hit_module = SimpleModule.regression(hit_model, metrics={'mae': MeanAbsoluteError()})
hit_logger = CSVLogger('logs', name= 'hitters')
hit_trainer = Trainer(deterministic=True, max_epochs=50, 
                        log_every_n_steps=5, 
                        logger= hit_logger,
                        callbacks=[ErrorTracker()])
hit_trainer.fit(hit_module, datamodule=hit_dm)
#Evaluate performance on our test data using test() method of our trainer
hit_trainer.test(hit_module, datamodule=hit_dm)
#Retrieve logged summaries
hit_results = pd.read_csv(hit_logger.experiment.metrics_file_path)

# Generic function to produce the plot
def summary_plot(results, ax, col='loss', valid_legend='Validation',
                    training_legend='Training', ylabel='Loss', fontsize=20):
    
    for (column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}']
                                        ['black', 'red'], [training_legend, valid_legend]):
        results.plot(x='epoch', y=column, label=label, marker='o', color=color, ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    return ax

fig,ax = subplots(1, 1, figsize=(6, 6))
ax= summary_plot(hit_results, ax, col='mae', ylabel='MAE', valid_legend='Validation(=Test)')
ax.set_ylim([0, 400])
ax.set_xticks(np.linspace(0, 50, 11).astype(int))
# Call eval() since it tells torch to effectively consider this model to be fitted so that we can 
# predict on new data
hit_model.eval()
preds = hit_module(X_test_t)
torch.abs(Y_test_t - preds).mean()
# We delete all references to the torch objects to ensure these processes will be killed 
del(Hitters, hit_model, hit_dm, hit_logger, hit_test, hit_train, X, Y, X_test, X_train, 
    Y_train, Y_test, X_test_t, Y_test_t, hit_trainer, hit_module)


#2. Multilayer Network on the MNIST Digit Data
(mnist_train, mnist_test) = [MNIST(root='data',train=train, download= True, transform=ToTensor())
                                for train in [True, False]]
mnist_dm = SimpleDataModule(mnist_train, mnist_test, validation = 0.2, num_workers = max_num_workers, 
                            batch_size= 256)
for idx, (X_, Y_) in enumerate(mnist_dm.train_dataloader()) :
    #print('X: ', X_.shape)
    #print('Y: ', Y_.shape)
    if idx >=1:
        break
# Now we specify our neural network
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(0.4))
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3))
        self._forward = nn.Sequential(
            self.layer1, self.layer2, nn.Linear(128, 10))
    
    def forward(self, x):
        return self._forward(x)
mnist_model = MNISTModel()
summary(mnist_model, input_data = X_, col_names= ['input_size', 'output_size', 'num_params'])
mnist_module = SimpleModule.classification(mnist_model)
mnist_logger = CSVLogger('logs', name= 'MNIST')
mnist_trainer = Trainer(deterministic= True, max_epochs= 30, logger= mnist_logger, callbacks=[ErrorTracker()])
mnist_trainer.fit(mnist_module, datamodule= mnist_dm)
mnist_results = pd.read_csv(mnist_logger.experiment.metrics_file_path)
fig, ax = subplots(1, 1, figsize = (6,6))
summary_plot(mnist_results, ax, col= 'accuracy', ylabel= 'Accuracy')
ax.set_ylim([0.5, 1])
ax.set_ylabel('Accuracy')
ax.set_xticks(np.linspace(0, 30, 7).astype(int))
mnist_trainer.test(mnist_module, datamodule= mnist_dm)

#3. Convolutional Neural Network
(cifar_train, cifar_test) = [CIFAR100(root= 'data', train=train, download= True)
                                for train in [True, False]]
transform = ToTensor()
cifar_train_X = torch.stack([transform(x) for x in cifar_train.data])
cifar_test_X= torch.stack([transform(x) for x in cifar_test.data])
cifar_train = TensorDataset(cifar_train_X, torch.tensor(cifar_train.targets))
cifar_test = TensorDataset(cifar_test_X, torch.tensor(cifar_test.targets))
#Create a data module
cifar_dm= SimpleDataModule(cifar_train, cifar_test, validation= 0.2, num_workers= max_num_workers, 
                            batch_size= 128)
#Look at the shape of typical batches in our data loaders 
for idx, (X_, Y_) in enumerate(cifar_dm.train_dataloader()) :
    #print('X: ', X_.shape)
    #print('Y: ', Y_.shape)
    if idx >= 1:
        break

fig, axes = subplots(5, 5, figsize=(10, 10))
rng = np.random.default_rng(4)
indices = rng.choice(np.arange(len(cifar_train)), 25, replace= False).reshape((5, 5))
for i in range(5) :
    for j in range(5):
        idx = indices[i, j]
        axes[i, j].imshow(np.transpose(cifar_train[idx][0], [1,2,0]), interpolation= None)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

class BuldingBlock(nn.Module) :
    def __init__(self, in_channels, out_channels):
        super(BuldingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size= (3, 3), 
                                padding= 'same')
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size= (2,2))
    
    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))

class CIFARModel(nn.Module) :
    def __init__(self):
        super(CIFARModel, self). __init__()
        sizes = [(3, 32), (32, 64), (64, 128), (128, 256)]
        self.conv = nn.Sequential(*[BuldingBlock(in_, out_)
                                    for in_, out_ in sizes])
        self.output = nn.Sequential(nn.Dropout(0.5), nn.Linear(2*2*256, 512),
                                    nn.ReLU(), nn.Linear(512, 100))
    
    def forward(self, x):
        val = self.conv(x)
        val = torch.flatten(val, start_dim= 1)
        return self.output(val)

cifar_model = CIFARModel()
summary(cifar_model, input_data= X_, col_names = ['input_size', 'output_size', 'num_params'])

cifar_optimizer = RMSprop(cifar_model.parameters(), lr= 0.001)
cifar_module = SimpleModule.classification(cifar_model, optimizer= cifar_optimizer)
cifar_logger = CSVLogger('logs', name= 'CIFAR100')
cifar_trainer = Trainer(deterministic= True, max_epochs= 30, logger= cifar_logger, 
                        callbacks=[ErrorTracker()])
cifar_trainer.fit(cifar_module, datamodule= cifar_dm)

log_path = cifar_logger.experiment.metrics_file_path
cifar_results = pd.read_csv(log_path)
fig, ax = subplots(1, 1, figsize=(6, 6))
summary_plot(cifar_results, ax, col= 'accuracy', ylabel= 'Accuracy')
ax.set_xticks(np.linspace(0, 10, 6).astype(int))
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
cifar_trainer.test(cifar_module, datamodule= cifar_dm)
#Hardware acceleration
try:
    for name, metric in cifar_module.metrics.items():
        cifar_module.metrics[name] = metric.to('mps')
    cifar_trainer_mps = Trainer(accelerator = 'mps', deterministic= True, max_epochs= 30)
    cifar_trainer_mps.fit(cifar_module, datamodule= cifar_dm)
    cifar_trainer_mps.test(cifar_module, datamodule= cifar_dm)
except:
    pass


#4. Using Pretrained CNN Models
resize = Resize((232, 232))
crop = CenterCrop(224)
normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
imgfiles = sorted([f for f in glob('book_images/*')])
