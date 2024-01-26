# Rainfall_prediction_with_overNN

To import our models in Python, please follow the steps outlined below:

1. Locate Model.py in your working directory and import it in python.
```
import torch
from Model import Rainfall_linearnet
```
2. Specify a model you want to import.
```
region= 'WP' # 'WP' or 'EP'
rain_type = 'sf' # 'sf' for stratiform, 'dp' for deep convective, 'sh' for shallow convective.
L=5
W=600
layers = [163]+[W]*(L-1)+[1]
activation='Tanh'
```
3. Load the model. Don't forget to put the model in the working directory.
```
model = Rainfall_linearnet(L, layers, dropout_p=.0, net_type='r', activation=activation)
model_to_load = [region, rain_type, L, W, activation]
model_to_load = '_'.join([str(x) for x in model_to_load])
model_to_load += '.pt'
model.load_state_dict(torch.load(model_to_load, map_location=device))
model.eval()
```
