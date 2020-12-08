from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def ModelSelector(training_data:list, testing_data:list, model, max_iterations = 10000,
                   max_num_hidden_layers = 4, min_num_hidden_layers = 1,
                   max_nodes_per_layer = 256, min_nodes_per_layer = 16, loss_func = 'any', activation = 'any',
                   learning_rate = 'any', min_learning_rate = 0.0001):
    '''
    :param training_data: [train_X, train_y]
    :param testing_data: [test_X, test_y]
    :param model: {'NNClassifier', 'NNRegressor'}
    :param max_iterations: Maximum no. epochs for the neural network
    :param max_num_hidden_layers: Maximum no. of hidden layers allowed for the model
    :param min_num_hiddenLayers: Minimum no. of hidden layers needed for the model
    :param max_nodes_per_layer: Maximum no. of neurons allowed per layer
    :param min_nodes_per_layer: Minimum no. of neurons needed per layer
    :param loss_func: {'lbfgs', 'sgd', 'adam', 'any'}
    :param activation: {'relu', 'tanh', 'logistic', 'identity', 'any'}
    :param learning_rate: {'constant', 'invscaling', 'adaptive', 'any'}
    :param min_learning_rate: If the learning rate after a certain epoch is less than the specified min_learning_rate, maximum optimisation for that model
     considered to be achieved
    :return: Best model
    '''

    try:
        train_X, train_y = training_data[0], training_data[1]
        test_X, test_y = testing_data[0], testing_data[1]
    except:
        raise ValueError("Training and testing data must be list!")

    nodes = []
    for t in range(min_nodes_per_layer, max_nodes_per_layer + 1):
        if t%2 == 0:
            nodes.append(t)

    if loss_func == 'any':
        loss_func = ['adam', 'sgd', 'lbfgs']
    elif loss_func in {'lbfgs', 'sgd', 'adam', 'any'}:
        loss_func = [loss_func,]
    else:
        raise ValueError(f"loss_func must be adam, sgd, lbfgs or any, not {loss_func}!")
    if activation == 'any':
        activation = ['relu', 'tanh', 'logistic', 'identity']
    elif activation in {'relu', 'tanh', 'logistic', 'identity', 'any'}:
        activation = [activation,]
    else:
        raise ValueError(f"activation must be relu, tanh, logistic, identity, or any, not {activation}")
    if learning_rate == 'any':
        learning_rate = ['constant', 'invscaling', 'adaptive']
    elif learning_rate in {'constant', 'invscaling', 'adaptive', 'any'}:
        learning_rate = [learning_rate,]
    else:
        raise ValueError(f"learnin_rate must be constant, invscaling, adaptive or any, not {learning_rate}")

    combos = []
    for c in range(len(nodes)):
        for b in range(min_num_hidden_layers, max_num_hidden_layers + 1):
            layers = []
            for i in range(b):
                if i + c < len(nodes):
                    layers.append(nodes[i + c])
                else:
                    layers.append((nodes[(i + c) - len(nodes)]))
            combos.append(layers)

    num_models = len(combos) * len(loss_func) * len(activation) * len(learning_rate)
    models = [-4]
    scores = [-4]
    to_train = []
    print(f'Training: {num_models} models to train')
    for loss_function in loss_func:
        for activation_function in activation:
            for lr in learning_rate:
                for hidden_layers in combos:
                    to_train.append({"hidden_layer_sizes": hidden_layers,
                    "activation": activation_function,
                    "solver": loss_function,
                    "learning_rate": lr,
                    "tol": min_learning_rate,
                    "max_iter": len(hidden_layers) * max_iterations})
	
    for data in tqdm(to_train):
        if model == 'NNClassifier':
            mlp = MLPClassifier(**data)
            mlp.fit(train_X, train_y)
            accuracy = accuracy_score(mlp.predict(test_X), test_y)
            if accuracy > max(scores):
                        models.append(mlp)
                        scores.append(accuracy)
                    
    return models[scores.index(max(scores))]

#testing purposes
if __name__ == "__main__":
    print(ModelSelector(training_data=[[[0, 1], [1, 3], [2, 4], [3, 2], [4, 4]], [1, 4, 6, 5, 8]], testing_data=[[[3, 2], [9, 2]],[5, 11]], model = 'NNClassifier', max_nodes_per_layer=16, min_nodes_per_layer=2, min_num_hidden_layers=1, max_num_hidden_layers=3))