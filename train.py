import tensorflow as tf
from model import SuperbModel, SuperbLayer
from train_utils import random_batch, print_status_bar, get_train_step, train_model
from hyperparameter_search import change_hyperparameters, guided_hyperparameter_search

mnist = tf.keras.datasets.mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
X_train_full, X_test = X_train_full / 255, X_test / 255	

if __name__ == "__main__":    
    guided_hyperparameter_search(X_train_full, y_train_full, patience=5)

 
