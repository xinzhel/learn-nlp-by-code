I found that `AllenNLP` is a kinda unique NLP tool to be implemented fundamentally via a logic combining typical NLP with deep learning .
# AllenNLP: Analyzing its mysterious allennlp commands and JSON files
AllenNLP provides an agile experimental tool via only **JSON configuration (or config) files** like [this one](https://github.com/allenai/allennlp-models/blob/main/training_config/classification/basic_stanford_sentiment_treebank.jsonnet) and its **commands (subcommands)**. However, as a programmer, I am not happy when many implementations are unknown.

Therefore, to ensure I have clearly analyzed the mysteries behind allennlp commands and JSON files in this post, I set two practical goals:
1. use customized python script rather than `AllenNLP` commands and subcommands: To achieve this, I need to explore: **How `AllenNLP` commands and subcommands work (i.e., `allennlp train ...` for training)?**. At the end, [this simple, more transparent python script](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py) is generated to achieve the same behaviour of `allennlp train` command.
2. construct all the objects by myself for the training process: To achieve this, I need to explore: **How the json config file is parsed?**. Note that the above goal just explains how command line arguments are parsed, but the mystery of parsing JSON files into Python objects and organizing them for training is still unknown in the class `TrainModel` in [this simple, more transparent python script](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py). 

BTW, for those reading this post, I recommend trying to use provided debug codes in [my github](https://github.com/xinzhel/allennlp-code-analysis) for fully understanding since I don't think only my writing is good enough for most of people to easily understand the complicated allennlp design (I use it for more than half an year but still not fully understand).


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

Therefore, to replicate the behaviour of the command (`allennlp`) and subcommand (`train`) in a python script, I only need to implement two functions by myself in the [Python scirpt](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py):
1. the argument parsing function: this correponds to `allennlp.commands.train.Trainadd_subparser`.
2. the train function: this corresponds to `allennlp.commands.train.train_model_from_file`. Since the detail of this function relates to the next topic, I directly call the function and leave the rewriting work as the goal for the next challenge.


##  How the JSON config file is parsed?
Following the [Python script](https://github.com/xinzhel/allennlp-code-analysis/blob/master/scripts/main_clean_train.py), now, `train_model_from_file` could be further explored to see the parsing of [a JSON file](https://github.com/allenai/allennlp-models/blob/main/training_config/classification/basic_stanford_sentiment_treebank.jsonnet) specified for training a model. See the comment and code.

```
def train_model_from_file(...):
    # this would parse the JSON file to a dictionary
    params = Params.from_file(parameter_filename, overrides)

    # FINALLY, this would construct all objects according to the information in JSON file. The detail is shown in the following subsection.
    train_loop = TrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=0,
    )
    metrics = train_loop.run()
    train_loop.finish(metrics)
    model = train_loop.model
    return model
```


### from_params: from the JSON file to Python objects
This is implemented with the class `FromParams` which is inherited by all the classes in `allennlp`, specifically, the `FromParams.from_params` method. For example, we could easily construct the model defined below via a JSON dictionary `{"input_size": 64, "output_size": 2}`.

```
from allennlp.common import FromParams
from allennlp.common import Params
import json

class Model(FromParams):
    def __init__(self, input_size: int, output_size: int):
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size
        

#  params is a dictionary constructed by reading the JSON file in `file_path`
params = Params(json.loads(file_path))
model = Model.from_params(params)
```

Since this part has been explained very well in the [official guide], I just summarize some points to benefit further exploration.
* JSON key parsing: match class argument names of objects to be constructed.
* JSON value parsing: type annotation (e.g., int) is used to parse values from JSON into correct data types of class arguments for objects to be constructed. 
* Recursively parsing: If an argument of `Model` are another `Model` object or any object requiring arguments, this could be defined by nested JSON dictionary. Of course, the object has to be constructed from the class inheriting `FromParam`. This actually limits us to directly use Pytorch code if we want to benefit from JSON definitions.

Now, we know where ` TrainModel.from_params` comes from to parse the JSON file. If we want to see immplementation of parsing, we could look at the [source code of `FromParams`](https://github.com/allenai/allennlp/blob/5338bd8b4a7492e003528fe607210d2acc2219f5/allennlp/common/from_params.py#L558). However, once we look at `TrainModel` class, it inherits from `Registrable` class. Next, I'll explain this.




### Registrable: registration design
I think, if we have workflows containing many unstructured objects, this idea would lead to messy json files. But objects required for deep learning workflows (e.g., training) tend to have common operations and could be collected into a few abstract classes (e.g., `Model`, `DataReader`). This idea is actually one of foundamental objected-oriented priciples: polymorphism where **abstract base classes** encapsulate common operations, and **concrete subclasses** handle low-level details of data processing or model operations. In order to make **abstract base classes** have the ability to create objects of **concrete subclasses**, `allennlp` uses registration design which is implemented as the `Registrable` base class. The code is as below.

```
class Registrable(FromParams):
   
    _registry: ClassVar[DefaultDict[type, _SubclassRegistry]] = defaultdict(dict)

    default_implementation: Optional[str] = None
    
     @classmethod
    def register(
        cls, name: str, constructor: Optional[str] = None, exist_ok: bool = False
    ) -> Callable[[Type[_T]], Type[_T]]:
        ...
```
* 1st line: An abstract class inherited from the `Registrable` class could access all its concrete subclasses in the named registry `_registry`.
* 2nd line: If the JSON dictionary does not specify the concrete subclass via the 'type' key, `default_implementation` would be used.
* 3rd to 5th lines: To be able to access all the subclasses, we need to decorate all the subclasses via `@BaseClass.register(name)` (e.g., @Model.register("my_model_name")) or manually call `@BaseClass.register(name)(class)`. the argument `constructor` is to define the class method to create the object. I guess this is used by `from_params` method to construct objects. 

Let's manually register a default implementation for the `Model` class defined above.
```
from allennlp.common import FromParams

class Model(FromParams):
    default_implementation = "xxx"
    def __init__(self, input_size: int, output_size: int):
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size
Model.register("xxx")(LSTM)
```

Now, the default implementation is `LSTM`, and I name it as 'xxx' just to show that any names would be acceptable. Of course, we can use decorator.

```
Model.register("xxx")
class LSTM(Model):
    def __init(self, ...):
        super().init(...)
    
```

I think that the understanding is enough for me to implement the functionality of `TrainModel` by myself. 

First, in order to find where `TrainModel` constructs objects (e.g., `Model`, `DataReader`), we need to find how it is constructed. Since it inherits`Registrable`, we need to find the default constructor of `TrainModel`. The code is shown below. 
```
TrainModel.register("default", constructor="from_partial_objects")(TrainModel)
```
In the `from_params` method, one line as below could be found to fetch the rigistry of subclasses.
```
registered_subclasses = Registrable._registry.get(cls)
# if cls is TrainModel, `registered_subclasses` would be: {'default': (<class 'allennlp....TrainModel'>, 'from_partial_objects')}
```

`TrainModel` registers itself as the default implementation and constructs an object using `from_partial_object`. When we look at this class method, we could find the basic objects are constructed according to the params from JSON files (e.g., `serialization_dir`) and command line arguments (e.g., `serialization_dir`). The following code is easy to understand for anyone familiar with object-oriented programming.





## Reference
[The offical guide](https://guide.allennlp.org/using-config-files).


