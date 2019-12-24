import numpy as np
from knn import KNN

#F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    tp = sum([x == 1 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fp = sum([x == 0 and y == 1 for x, y in zip(real_labels, predicted_labels)])
    fn = sum([x == 1 and y == 0 for x, y in zip(real_labels, predicted_labels)])
    if(tp+fp ==0 or tp + fn == 0):
        return 0
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    if(precision + recall == 0):
        return 0
    f1 = (2* precision*recall)/(precision+recall)
    return f1


class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        dimensions = len(point1)
        for dimension in range(dimensions):
            sum = sum + (abs(point2[dimension] - point1[dimension]))**3
        minkowski_dist = sum**(1/3)
        return minkowski_dist

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0;
        dimensions = len(point1)
        for dimension in range(dimensions):
            sum = sum + (point1[dimension] - point2[dimension])**2
        euclidean_dist = sum**(0.5)
        return euclidean_dist

    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        dimensions = len(point1)
        for dimension in range(dimensions):
            sum = sum + point1[dimension]*point2[dimension]
        return sum

    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        point1_mag = 0
        point2_mag = 0
        dimensions = len(point1)
        sum = Distances.inner_product_distance(point1, point2)
        for dimension in range(dimensions):
            point1_mag = point1_mag + point1[dimension]**2
            point2_mag = point2_mag + point2[dimension]**2

        point1_mag = point1_mag**(1/2)
        point2_mag = point2_mag**(1/2)
        sum = sum/(point1_mag*point2_mag)
        return 1 - sum

    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        sum = 0
        dimensions = len(point1)
        for dimension in range(dimensions):
            sum = sum + (point1[dimension] - point2[dimension])**2
        sum = -np.exp(-0.5 * sum)
        return sum


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        import pdb
        max_f1 = 0
        max_k = 1
        max_dist_func = None
        max_knn= None
        for k in range(1, min(len(x_train),30), 2):
            for dist_fun in distance_funcs:
                knn = KNN(k, distance_funcs[dist_fun])
                knn.train(x_train, y_train)
                predicted_labels = knn.predict(x_val)
                f1 = f1_score(y_val, predicted_labels)
                if(f1 > max_f1):
                    max_f1 = f1
                    max_k = k
                    max_dist_func = dist_fun
                    max_knn = knn
                elif(f1 == max_f1 and list(distance_funcs.keys()).index(dist_fun) < list(distance_funcs.keys()).index(max_dist_func)):
                    max_f1 = f1
                    max_k = k
                    max_dist_func = dist_fun
                    max_knn = knn
                elif(f1 == max_f1 and list(distance_funcs.keys()).index(dist_fun) == list(distance_funcs.keys()).index(max_dist_func) and k < max_k):
                    max_f1 = f1
                    max_k = k
                    max_dist_func = dist_fun
                    max_knn = knn

        self.best_k = max_k
        self.best_distance_function = distance_funcs[max_dist_func]
        self.best_model = max_knn

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        max_f1 = -100
        max_k = 1
        max_dist_func = None
        max_scaler = None
        max_knn= None
        for k in range(1, min(len(x_train),30), 2):
            for dist_fun in distance_funcs:
                for scaler in scaling_classes:
                    knn = KNN(k, distance_funcs[dist_fun])
                    scaler_ = scaling_classes[scaler]()
                    scaled_x = scaler_(x_train)

                    knn.train(scaled_x, y_train)
                    predicted_labels = knn.predict(scaler_(x_val))
                    f1 = f1_score(y_val, predicted_labels)
                    if(f1 > max_f1):
                        max_f1 = f1
                        max_k = k
                        max_scaler = scaler
                        max_dist_func = dist_fun
                        max_knn = knn
                    elif(f1 == max_f1):
                        if(list(scaling_classes.keys()).index(scaler) < list(scaling_classes.keys()).index(max_scaler)):
                            max_f1 = f1
                            max_k = k
                            max_scaler = scaler
                            max_dist_func = dist_fun
                            max_knn = knn
                    elif(f1 == max_f1):
                        if(list(scaling_classes.keys()).index(scaler) == list(scaling_classes.keys()).index(max_scaler)):
                            if(list(distance_funcs.keys()).index(dist_fun) < list(distance_funcs.keys()).index(max_dist_func)):
                                max_f1 = f1
                                max_k = k
                                max_scaler = scaler
                                max_dist_func = dist_fun
                                max_knn = knn
                    elif(f1 == max_f1):
                        if(list(scaling_classes.keys()).index(scaler) == list(scaling_classes.keys()).index(max_scaler)):
                            if(list(distance_funcs.keys()).index(dist_fun) == list(distance_funcs.keys()).index(max_dist_func) and k < max_k):
                                max_f1 = f1
                                max_k = k
                                max_scaler = scaler
                                max_dist_func = dist_fun
                                max_knn = knn
        self.best_k = max_k
        self.best_distance_function = max_dist_func
        self.best_scaler = max_scaler
        self.best_model = knn


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalised = []
        for point in features:
            if all(x == 0 for x in point):
                normalised.append(point)
            else:
                deno = np.sqrt(Distances.inner_product_distance(point, point))
                normalised_point = [x/deno for x in point]
                normalised.append(normalised_point)
        return normalised




class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.min=None
        self.max=None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """

        if self.min is None and self.max is None:
            self.max = np.amax(features, axis=0).tolist()
            self.min = np.amin(features, axis=0).tolist()
        normalised_features = []
        features = features.tolist()
        for point in range(len(features)):
            normalised_point=[]
            for i in range (len(features[point])):
                if(self.max[i] - self.min[i] == 0):
                    normalised_point.append(0)
                else:
                    scaled = features[point][i] - self.min[i]/(self.max[i] - self.min[i])
                    normalised_point.append(scaled)
            normalised_features.append(normalised_point)
        return normalised_features

