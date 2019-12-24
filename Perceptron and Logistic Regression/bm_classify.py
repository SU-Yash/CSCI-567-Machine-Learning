import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """

    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    transformed_y = np.where(y == 0, -1, 1)

    if loss == "perceptron":
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            loss = np.int32(transformed_y * (X.dot(w) + b) <= 0) * transformed_y
            update = loss.T.dot(X)
            w += step_size/N *update
            b += step_size/N *np.sum(loss)

    elif loss == "logistic":
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            loss = sigmoid(-transformed_y * (X.dot(w) + b)) * transformed_y
            update = loss.T.dot(X)
            w += step_size/N *update
            b += step_size/N *np.sum(loss)

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    #          Compute value                   #
    value = z
    value = 1/(1+np.exp(-z))
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = np.int32((X.dot(w) + b) > 0)

    elif loss == "logistic":
        #          Compute preds                   #
        preds = np.zeros(N)
        preds = np.int32((X.dot(w) + b) > 0)

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent
    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0


    np.random.seed(42)
    if gd_type == "sgd":
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        for it in range(max_iterations):
            random_point = np.random.choice(N)
            xn = X[random_point]
            yn = y[random_point]

            loss = xn.dot(w.T) + b  # 1*C
            loss -= loss.max()

            y_probs = np.exp(loss)
            y_probs /= y_probs.sum()

            y_probs[yn] -= 1
            update = np.dot(y_probs.reshape(C, 1), xn.reshape(1, D))
            w -= step_size * update
            b -= step_size * y_probs
        

    elif gd_type == "gd":
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)

        for it in range(max_iterations):
            loss = X.dot(w.T) + b  # N*C

            y_probs = np.exp(loss)
            y_probs /= y_probs.sum(axis=1, keepdims=True)

            onehot = np.zeros([N, C])
            onehot[np.arange(N), y.astype(int)] = 1.0

            err = y_probs - onehot
            update = np.dot(err.T, X)
            w -= step_size/N * update
            b -= step_size/N * err.sum(axis=0)
  
    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns: 
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #          Compute preds                   #
    preds = X.dot(w.T) + b  # N*C
    preds = np.argmax(preds, axis=1)

    assert preds.shape == (N,)
    return preds



        