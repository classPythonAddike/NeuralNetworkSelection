# NeuralNetworkSelector for Scikit-learn

  - Use this module to get the most accurate neural network

### Installation

```sh
> pip install NeuralNetworkSelector
```

### Usage

```sh
from NeuralNetworkSelector import BestModel
. . . .
model = BestModel([train_X, train_y], [test_X, test_y], model = "NNRegressor")
model.predict(. . . .)
```