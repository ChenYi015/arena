#
# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import json

import tensorflow as tf


def print_env():
    """Print environment information and distributed training configuration."""
    info = {"PID": os.getpid()}

    # Check for distributed training environment variables.
    env_vars = ["TF_CONFIG", "MASTER_ADDR", "MASTER_PORT"]

    for var in env_vars:
        if var in os.environ:
            info[var] = json.loads(
                os.environ[var]) if var == "TF_CONFIG" else os.environ[var]

    print(info)


def get_tf_config_info():
    """Parse TF_CONFIG to determine training topology."""
    if "TF_CONFIG" not in os.environ:
        return None, None, None

    tf_config = json.loads(os.environ["TF_CONFIG"])
    cluster = tf_config.get("cluster", {})
    task = tf_config.get("task", {})

    has_ps = "ps" in cluster and len(cluster["ps"]) > 0
    task_type = task.get("type", "worker")
    task_index = task.get("index", 0)

    return has_ps, task_type, task_index


def main():
    # Parse TF_CONFIG to determine training topology
    has_ps, task_type, task_index = get_tf_config_info()

    # For PS-Worker mode, parameter servers need to start a server and wait
    if has_ps and task_type == "ps":
        print_env()
        print(f"Task type: {task_type}, Task index: {task_index}")
        print("This is a parameter server. Starting server...")

        # Create cluster resolver and start server
        cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
        server = tf.distribute.Server(
            cluster_resolver.cluster_spec(),
            job_name=task_type,
            task_index=task_index,
            protocol="grpc"
        )
        print(
            f"Parameter server {task_index} started, waiting for termination...")
        server.join()
        return

    parser = argparse.ArgumentParser(description="TensorFlow MNIST Example")
    parser.add_argument(
        "--data",
        default="../data",
        metavar="D",
        help="directory where MNIST data is cached",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="learning rate decay factor (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="save the trained model to disk",
    )
    parser.add_argument(
        "--dir",
        default=os.path.join(os.path.dirname(__file__), "logs"),
        metavar="L",
        help="directory where TensorBoard logs are stored",
    )
    args = parser.parse_args()
    print_env()

    # Setup distributed strategy based on TF_CONFIG topology.
    use_ps_strategy = False
    if has_ps is not None:  # TF_CONFIG is set
        if has_ps:
            # PS-Worker distributed training mode (worker node).
            print(f"Task type: {task_type}, Task index: {task_index}")
            print("Using ParameterServerStrategy for PS-Worker mode.")
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            strategy = tf.distribute.experimental.ParameterServerStrategy(
                cluster_resolver)
            use_ps_strategy = True
            print(
                f"ParameterServerStrategy initialized for worker {task_index}.")
        else:
            # Multi-worker training without parameter servers (all workers).
            print("Using MultiWorkerMirroredStrategy for multi-worker mode.")
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            print(
                f"Using MultiWorkerMirroredStrategy with {strategy.num_replicas_in_sync} replicas.")
    elif not args.no_cuda and len(tf.config.list_physical_devices("GPU")) > 0:
        # Single-worker multi-GPU training.
        strategy = tf.distribute.MirroredStrategy()
        print(
            f"Using MirroredStrategy with {strategy.num_replicas_in_sync} GPUs.")
    else:
        # Single-device training (CPU or single GPU).
        print("Using single device training.")
        strategy = tf.distribute.get_strategy()

    tf.random.set_seed(args.seed)

    # Load MNIST dataset.
    (x_train, y_train), (x_test,
                         y_test) = tf.keras.datasets.mnist.load_data(path=args.data)
    train_size, test_size = len(x_train), len(x_test)

    # Dataset creation function for PS-Worker mode
    def make_dataset(is_training=True):
        if is_training:
            dataset = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32) / 255.0, -1), y))
                .shuffle(train_size)
                .batch(args.batch_size)
            )
        else:
            dataset = (
                tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32) / 255.0, -1), y))
                .batch(args.test_batch_size)
            )
        return dataset

    # For non-PS modes, create datasets directly
    if not use_ps_strategy:
        train_dataset = make_dataset(is_training=True)
        test_dataset = make_dataset(is_training=False)

    # Create and compile model within strategy scope for distributed training.
    with strategy.scope():
        # Build CNN model.
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ])

        # Compile model with loss, optimizer, and metrics.
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=args.lr),
            metrics=["accuracy"],
        )

    # Setup training callbacks.
    callbacks = []

    # TensorBoard callback for visualization.
    callbacks.append(
        tf.keras.callbacks.TensorBoard(log_dir=args.dir, histogram_freq=1)
    )
    # Learning rate scheduler callback.
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * args.gamma
        )
    )

    # Custom callback for detailed progress logging.
    class LoggingCallback(tf.keras.callbacks.Callback):
        """Custom callback to log training progress and test accuracy."""

        def __init__(self, log_interval, train_size, test_size, batch_size):
            super().__init__()
            self.log_interval = log_interval
            self.train_size = train_size
            self.test_size = test_size
            self.batch_size = batch_size
            self.batch_count = 0
            self.epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch
            self.batch_count = 0

        def on_train_batch_end(self, batch, logs=None):
            self.batch_count += 1
            if self.batch_count % self.log_interval == 0:
                samples_processed = min(
                    self.batch_count * self.batch_size, self.train_size)
                percentage = min(100.0 * samples_processed /
                                 self.train_size, 100.0)
                print(
                    f"Train Epoch: {self.epoch + 1} "
                    f"[{samples_processed}/{self.train_size} "
                    f"({percentage:.0f}%)]\t"
                    f"Loss: {logs['loss']:.6f}"
                )

        def on_test_end(self, logs=None):
            accuracy = logs["accuracy"]
            correct = int(accuracy * self.test_size)
            print(
                f"\nTest Accuracy: {correct}/{self.test_size} "
                f"({accuracy * 100.0:.2f}%)\n"
            )

    logging_callback = LoggingCallback(
        args.log_interval, train_size, test_size, args.batch_size)
    callbacks.append(logging_callback)

    # Early stopping for dry run.
    if args.dry_run:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="loss", patience=0)
        )

    # Train the model.
    if use_ps_strategy:
        # PS-Worker training: use ClusterCoordinator API
        coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)

        # Create distributed dataset - use a simple callable that returns a dataset
        def dataset_fn():
            dataset = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .map(lambda x, y: (tf.expand_dims(tf.cast(x, tf.float32) / 255.0, -1), y))
                .shuffle(train_size)
                .batch(args.batch_size)
                .repeat()
            )
            return dataset

        per_worker_dataset = coordinator.create_per_worker_dataset(dataset_fn)

        # Define training step - receives the distributed dataset directly
        @tf.function
        def per_worker_step_fn(iterator):
            def step_fn(batch):
                x, y = batch
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)
                    per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y, predictions, from_logits=True)
                    loss = tf.nn.compute_average_loss(per_example_loss)
                gradients = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                return loss

            per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # Train with coordinator
        print("Starting PS-Worker distributed training...")
        steps_per_epoch = train_size // args.batch_size

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            total_loss = 0.0
            num_logged = 0

            for step in range(steps_per_epoch):
                result = coordinator.schedule(
                    per_worker_step_fn, args=(per_worker_dataset,))

                if (step + 1) % args.log_interval == 0:
                    coordinator.join()
                    loss_value = result.fetch()
                    total_loss += loss_value
                    num_logged += 1
                    samples_processed = min(
                        (step + 1) * args.batch_size, train_size)
                    percentage = min(
                        100.0 * samples_processed / train_size, 100.0)
                    print(
                        f"Train Epoch: {epoch + 1} "
                        f"[{samples_processed}/{train_size} ({percentage:.0f}%)]\t"
                        f"Loss: {loss_value:.6f}"
                    )

            coordinator.join()
            avg_loss = total_loss / max(num_logged, 1)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.6f}")

            # Apply learning rate decay
            current_lr = model.optimizer.learning_rate.numpy()
            model.optimizer.learning_rate.assign(current_lr * args.gamma)

            if args.dry_run:
                break

        print("PS-Worker training completed.")
    else:
        # MultiWorkerMirrored or single device training
        model.fit(
            train_dataset,
            epochs=args.epochs,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=2 if not args.dry_run else 1
        )

    # Save the model.
    if args.save_model:
        model.save("mnist_cnn.h5")
        print("Model saved to mnist_cnn.h5")


if __name__ == "__main__":
    main()
