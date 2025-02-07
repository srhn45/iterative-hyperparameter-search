import tensorflow as tf
import numpy as np
import logging

def change_hyperparameters(best_params, search_space, change_factor=0.2, change_proba=0.5):
    """Change a subset of the best hyperparameters within a small range."""
    new_params = best_params.copy()
    for key in best_params.keys():
        if np.random.rand() < change_proba:  # Probability of changing parameter
            if isinstance(search_space[key][0], int):  # if integer parameter
                delta = max(1, int(change_factor * best_params[key]))
                new_params[key] = best_params[key] + np.random.randint(-delta, delta+1) # can constrain the search space with np.clip if necessary
            elif isinstance(search_space[key][0], float):  # if float parameter
                delta = change_factor * best_params[key]
                new_params[key] = best_params[key] + np.random.uniform(-delta, delta) # can constrain the search space with np.clip if necessary
            else:  # if categorical parameter
                new_params[key] = np.random.choice(search_space[key])
    return new_params

def guided_hyperparameter_search(X_train_full, y_train_full, max_trials=50, patience=10, initial_epochs=5, change_factor=0.2, change_proba=0.5, output_dim=10):
    import logging
    from model import SuperbModel, SuperbLayer
    from train_utils import random_batch, print_status_bar, get_train_step, train_model
    from sklearn.model_selection import train_test_split

    search_space = {
        "batch_size": [16, 64],
        "optimizer": ["Nadam", "Adam", "RMSprop", "SGD"],
        "n_layers": [3, 10],
        "n_neurons": [32, 128],
        "dropout_rate": [0.1, 0.3],
        "activation": ["relu", "swish"],
        "learning_rate": [1e-4, 1e-2],
        "decay_steps": [1e3, 1e6],
        "decay_rate": [0.85, 0.999]
    }

    best_acc = 0
    best_params = None
    no_improvement_count = 0
    n_epochs = initial_epochs

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__) # set up logging

    for trial in range(1, max_trials + 1):
        logger.info(f"\nTrial {trial}/{max_trials}")
        
        if trial == 1 or best_params is None:
            # random selection for first trial
            params = {key: np.random.choice(search_space[key]) if isinstance(search_space[key][0], str) 
                      else np.random.uniform(*search_space[key]) if isinstance(search_space[key][0], float)
                      else np.random.randint(search_space[key][0], search_space[key][1] + 1) 
                      for key in search_space}
        else:
            # perturb the best hyperparameters
            params = change_hyperparameters(best_params, search_space, change_factor, change_proba)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=params["learning_rate"],
            decay_steps=params["decay_steps"],
            decay_rate=params["decay_rate"],
            staircase=False
        )
        optimizer = getattr(tf.keras.optimizers, params["optimizer"])(learning_rate=lr_schedule)

        model = SuperbModel(n_layers=params["n_layers"], n_neurons=params["n_neurons"],
                            dropout_rate=params["dropout_rate"], activation=params["activation"], output_dim=output_dim)
        model.build(input_shape=(None, np.prod(X_train_full.shape[1:])))
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)
        valid_acc = train_model(model, X_train, y_train, X_valid, y_valid, optimizer, 
                                tf.keras.losses.SparseCategoricalCrossentropy(), params["batch_size"], n_epochs)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_params = params
            no_improvement_count = 0  # resetting the patience counter
            n_epochs += 2  # increases max epochs for next trials if it finds a better set of hyperparameters.
        else:
            no_improvement_count += 1

        logger.info(f"Best so far: {best_params} with accuracy {best_acc:.4f}")

        change_factor *= (1 - no_improvement_count / patience)  # adaptive search

        if no_improvement_count >= patience:
            logger.info("No improvement for several trials. Stopping search.")
            break

    return best_params, best_acc 
