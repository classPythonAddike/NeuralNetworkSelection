# NeuralNetworkSelector for Scikit-learn

  - Use this module to get the most accurate neural network

### Installation

```sh
> pip install NeuralNetworkSelector
```

### Usage

```python
from NeuralNetworkSelector import ModelSelector
. . . .
model = ModelSelector([train_X, train_y], [test_X, test_y], model = "NNRegressor")
model.predict(. . . .)
```