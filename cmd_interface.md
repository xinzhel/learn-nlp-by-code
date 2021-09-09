# AllenNLP: how to use config file explicitly via customized python script?
In order to use config file explicitly via customized python script, I wanna solve the following two questions:

1. How `AllenNLP` commands and subcommands work (i.e., `allennlp train ...` for training)? 

2. How the json config file is parsed?  

I found that answering the first question is enough to create customized python script. For the second question, I found that the whole design philoshophy is built for parsing json config file, and the offical guide has the answer for [this question](https://guide.allennlp.org/using-config-files#2).

## Analyzing how `allennlp train/predict/...` commmands work?
Here, I use `allennlp train` command as example.

**Where is the entry point?**

Firstly, I need to trace where the main command `allennlp` comes from. Since `allennlp` is recognized by bash terminal, it should be defined in the python bin path. Then, I did find it in `python/bin/allennlp`. This file is indeed a python file with shebang (#!) indicating the python path. When this file is runned, `allennlp/__main__.py` is indeed defined as the entry point with `if __name__ == "__main__":`. 

I define a python script which could be runned in a debugging mode to see the details step by step.

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

**How the subcommands (e.g., `train`) and arguments (e.g., config path) are parsed?**

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

**Creating the python script**
According to the analysis, I make the transparent and [simple python script](scripts/main_clean_train.py) which achieve the same behaviour of `allennlp train`.



## (Optional) AllenNLP registration design
All the subclasses are registered in the high-level abstract class, and they could be easilly assessed. For example, we could access the `allennlp.commands.train.Train` class via `Subcommand.by_name('train')` or `allennlp.commands.predict.Predict` class via `Subcommand.by_name('predict')`. It's kinda like a factory with all the templates (i.e., the subclasses), and you can create instances as many as you want from one place. The offical guide has discussed the usage of [registration](https://guide.allennlp.org/using-config-files#3).










