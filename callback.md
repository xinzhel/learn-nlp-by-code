# AllenNLP: training callback

Callback, generally, means calling a function or an object when the certain event happens.

AllenNLP create the general `TrainerCallback` object to make callback during the different stages of training. Each method in the object corresponds to acallback type. "Each one receives the state of the wrapper object as `self`. This enables easier state sharing between related callbacks."

* `on_backward`
* `on_batch`
* `on_epoch`
* `on_end`  

## `ConsoleLoggerCallback`

## `WandBCallback`
The following methods of `wandb` are used for `WandBCallback`.

* init()
* log() 
* watch()
* save() 
* finish()

* Histogram, 
* util.generate_id()