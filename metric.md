
# AllenNLP: how to record the training information via `metrics`?


`AllenNLP` provides almost all common metrics for experiments, e.g., average, auc, bleu.

During training (see `GradientDescentTrainer`), the metric would be updated at the end of each epoch. Validation metric would be used for early stopping which, by default, would be defined by loss. 

Below are the code snippets at the end of each batch training and epoch.

At the end of each batch training, it (1) updates the description; (2) updates callback; (3) updates tqdm.

```
# Update the description with the latest metrics
metrics = training_util.get_metrics(
    self.model,
    train_loss,
    train_reg_loss,
    batch_loss,
    batch_reg_loss,
    self._batches_in_epoch_completed,
)

for callback in self._callbacks:
    callback.on_batch(
        self,
        batch_group,
        batch_group_outputs,
        metrics,
        epoch,
        self._batches_in_epoch_completed,
        is_training=True,
        is_primary=self._primary,
        batch_grad_norm=batch_grad_norm,
    )

if self._primary:
    # Updating tqdm only for the primary as the trainers wouldn't have one
    description = training_util.description_from_metrics(metrics)
    batch_group_generator_tqdm.set_description(description, refresh=False)

if self._checkpointer is not None:
    self._checkpointer.maybe_save_checkpoint(
        self, self._epochs_completed, self._batches_in_epoch_completed
    )


```

At the end of each epoch, it calls `training_util.get_metrics()` to calculate the metrics with `train_loss`, `train_reg_loss` and `num_batches`. The returned `metrics` is a dictionary not only containing separate batch (regularized) loss, but also adding average (regularized) loss per batch and metrics defined/returned by the model.
```
metrics = training_util.get_metrics(
    self.model,
    train_loss,
    train_reg_loss,
    batch_loss=None,
    batch_reg_loss=None,
    num_batches=num_batches,
    reset=True,
)
```


