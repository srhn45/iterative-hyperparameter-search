import tensorflow as tf
import logging
import sys
import numpy as np


def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}" 
                          for m in [loss] + (metrics or [])])
    sys.stdout.write(f"\r{step}/{total} - {metrics}")
    sys.stdout.flush()

def get_train_step():
    @tf.function
    def train_step(inputs, labels, model, optimizer, loss_fn):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, predictions   
    return train_step

def train_model(model, X_train, y_train, X_valid, y_valid, optimizer, loss_fn, batch_size, n_epochs):
    """Train the model and return the validation accuracy."""
    import logging
    
    train_step = get_train_step()
    valid_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    mean_loss = tf.keras.metrics.Mean(name="mean_loss")
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__) # set up logging

    for epoch in range(1, n_epochs + 1):
        logger.info(f"Epoch {epoch}/{n_epochs}")
        n_steps = len(X_train) // batch_size
        for step in range(1, n_steps + 1):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size=batch_size)
            main_loss, y_pred = train_step(X_batch, y_batch, model, optimizer, loss_fn)
            mean_loss(main_loss)
            accuracy(y_batch, y_pred)
            print_status_bar(step, n_steps, mean_loss, [accuracy])

        valid_pred = model(X_valid, training=False)
        valid_metric.update_state(y_valid, valid_pred)
        logger.info(f"Validation Accuracy: {valid_metric.result().numpy()}")

        mean_loss.reset_state()
        accuracy.reset_state()

    return valid_metric.result().numpy() 
