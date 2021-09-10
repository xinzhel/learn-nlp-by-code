# AllenNLP: Analyzing its mysterious allennlp commands and JSON files
AllenNLP provides an agile experimental tool via only **JSON configuration (or config) files** and its **commands (subcommands)**. However, as a programmer, I am not happy when many implementations are unknown.

Therefore, to ensure I have clearly analyzed the mysteries behind allennlp commands and JSON files in this post, I set two practical goals:
1. use customized python script rather than `AllenNLP` commands and subcommands: To achieve this, I need to explore: **How `AllenNLP` commands and subcommands work (i.e., `allennlp train ...` for training)?**. At the end, [this simple, more transparent python script](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py) is generated to achieve the same behaviour of `allennlp train` command.
2. construct all the objects by myself for the training process: To achieve this, I need to explore: **How the json config file is parsed?**. Note that the above goal just explains how command line arguments are parsed, but the mystery of parsing JSON files into Python objects and organizing them for training is still unknown in the class `TrainModel` in this simple, more transparent python script](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py). 


## How `AllenNLP` commands and subcommands work?
Here, I use `allennlp train` command as example.

### Where is the entry point?
Firstly, I need to trace where the main command `allennlp` comes from. Since `allennlp` is recognized by bash terminal, it should be defined in the python bin path. Then, I did find it in `python/bin/allennlp`. This file is indeed a python file with shebang (#!) indicating the python path. When this file is runned, `allennlp/__main__.py` is indeed defined as the entry point with `if __name__ == "__main__":`. 

I define a python script which could be runed in a debugging mode to see the details step by step.

```
from allennlp.commands import main
import sys

config_file ='cofig_file.json' # the path of config file
serialization_dir = 'save_dir/' # the directory path for saving training information

# equal to running `allennlp train cofig_file.json  -s save_dir/ --include-package my_library` in the command line
sys.argv = [
    "allennlp",  # this would be useless since we directly call `main` method
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
]

main()
```

### How the subcommands (e.g., `train`) and arguments (e.g., config path) are parsed?
In `allennlp/__main__.py`, it calls `allennlp.commands.main()`, the subcommand and arguments ared processed here by calling `parser, args = parse_args(prog)`. `train` could be used as the subcommand due to the usage of `argparse` `subparsers`. The detail is explained with my simplified code snippet below.

```
# in `allennlp.commands.main`
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(title="Commands", metavar="")

# all the classes/functions/subcommands registered in `allennlp.Subcommand` (e.g., `allennlp.commands.train.Train`) would add a parser into subparsers. The detail of how the subcommands are registed in allennlp.Subcommand` would be discussed at the last.
# Now, the command line would understand `allennlp train` to use the 'train' subparser for `parser.parse_args()`
subparsers.add_parser('train', description=description, help="Train a model.")

# the `func` argument is set with the default function as `allennlp.commands.train.train_model_from_args`.
subparser.set_defaults(func=train_model_from_args)

# back to `allennlp.commands.main`
# except all the command line arguments (i.e., `sys.argv`)
args = parser.parse_args()

# call the function defined by `sys.argv[0]` (`train_model_from_args` here) with all parsed command line arguments
args.func(args)
```


##  How the JSON config file is parsed? 
The main design principle of `allennlp` is to construct all objects in `allennlp` with only the json file. [Here](https://github.com/allenai/allennlp-models/blob/main/training_config/classification/basic_stanford_sentiment_treebank.jsonnet) is a JSON file specified for training a classification model. This could be  I wanna fully understand how this JSON file works via the following analysis.

### from the JSON file to Python objectsI
This is implemented with ` `FromParams` which is inherited by all the classes in `allennlp`. For example, we could easily construct the model defined below via a JSON dictionary `{"input_size": 64, "output_size": 2}`.

```
from allennlp.common import FromParams

class Model(FromParams):
    def __init__(self, input_size: int, output_size: int):
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size
```
Specifically, this JSON dictionary could be read from a JSON file

```
from allennlp.common import Params
import json

params = Params(json.loads(file_path))
model = Model.from_params(params)
```

Since this part has been explained very well in the [official guide], I just summarize some points to benefit further exploration.
* JSON key parsing: match class argument names of objects to be constructed.
* JSON value parsing: type annotation (e.g., int) is used to parse values from JSON into correct data types of class arguments for objects to be constructed. 
* Recursively parsing: If an argument of `Model` are another `Model` object or any object requiring arguments, this could be defined by nested JSON dictionary. Of course, the object has to be constructed from the class inheriting `FromParam`. This actually limits us to directly use Pytorch code if we want to benefit from JSON definitions.

### 
I think, if we have workflows containing many unstructured objects, this idea would lead to messy json files. But objects required for deep learning workflows (e.g., training) tend to have common operations and could be collected into a few abstract classes (e.g., `Model`, `DataReader`). This idea is actually one of foundamental objected-oriented priciples: polymorphism where abstract base classes encapsulate common operations, and concrete instantiations handle low-level details of data processing or model operations.

## AllenNLP registration design

Specifically, `allennlp` uses `Registrable` 




`Registration`: decorator 

```
TrainModel.register("default", constructor="from_partial_objects")(TrainModel)
```

All the subclasses are registered in the high-level abstract class, and they could be easilly assessed. For example, we could access the `allennlp.commands.train.Train` class via `Subcommand.by_name('train')` or `allennlp.commands.predict.Predict` class via `Subcommand.by_name('predict')`. It's kinda like a factory with all the templates (i.e., the subclasses), and you can create instances as many as you want from one place. The official guide has discussed the usage of [registration](https://guide.allennlp.org/using-config-files#3).



**polymorphic dependency injection**





## Reference
[The offical guide](https://guide.allennlp.org/using-config-files).


