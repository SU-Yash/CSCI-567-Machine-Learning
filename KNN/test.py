from data import data_processing
from distance import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler, f1_score
from knn import KNN
import numpy as np

def main():
    distance_funcs = {
        'euclidean': Distances.euclidean_distance,
        'minkowski': Distances.minkowski_distance,
        'gaussian': Distances.gaussian_kernel_distance,
        'inner_prod': Distances.inner_product_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

    scaling_classes = {
        'min_max_scale': MinMaxScaler,
        'normalize': NormalizationScaler,
    }

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)
    print(list(distance_funcs.keys()).index('cosine_dist'))
    print(x_train[0])
    print(x_train[1])
    print(Distances.euclidean_distance(x_train[0], x_train[1]))
    print(Distances.minkowski_distance(x_train[0], x_train[1]))
    print(Distances.inner_product_distance(x_train[0], x_train[1]))
    print(Distances.cosine_similarity_distance(x_train[0], x_train[1]))
    print(Distances.gaussian_kernel_distance(x_train[0], x_train[1]))
    a = [[1], [4], [4], [10], [11], [23], [0], [50]]
    b = [[3, 4], [1, -1], [0, 0]]
    labels = [0, 0 ,0, 1, 1, 1, 0, 1]
    k = KNN(4, Distances.euclidean_distance)
    k.train(x_train, y_train)
    print(k.get_k_neighbors([2]))
    print(k.get_k_neighbors([28]))
    print(k.predict(x_val))

    tuner_without_scaling_obj = HyperparameterTuner()
    tuner_without_scaling_obj.tuning_without_scaling(distance_funcs, x_train, y_train, x_val, y_val)

    print("**Without Scaling**")
    print("k =", tuner_without_scaling_obj.best_k)
    print("distance function =", tuner_without_scaling_obj.best_distance_function)
    x = NormalizationScaler()

    tuner_with_scaling_obj = HyperparameterTuner()
    tuner_with_scaling_obj.tuning_with_scaling(distance_funcs, scaling_classes, x_train, y_train, x_val, y_val)

    print("\n**With Scaling**")
    print("k =", tuner_with_scaling_obj.best_k)
    print("distance function =", tuner_with_scaling_obj.best_distance_function)
    print("scaler =", tuner_with_scaling_obj.best_scaler)


if __name__ == '__main__':
    main()



