# How allennlp loads data via multiprocess?
https://github.com/allenai/allennlp/discussions/4933

## How different workers partition from the same data source?

# How trianing callbacks are used?
* Here is an example to build a comet-ml callback from scratch: https://www.pedro.ai/blog/2020/04/08/allennlp-callback-trainer-cometml/.
To log training statistics and metrics with wandb and tensorboard, it uses `LogWriterCallback` which `TensorBoardCallback` and `WandBCallback` are inherited from.

* How to use `WandBCallback`?: https://github.com/allenai/allennlp/discussions/5069


