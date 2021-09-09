# AllenNLP: command line interface




The `allennlp` command is defined in `python/bin`. When the command is called with arguments, it would run the this file `allennlp/allennlp/__main__.py` as the entry point which is actually call the `main` method in `allennlp.commands`.

The `main` method would first parse the arguments and run the subcommands (e.g., `train`, `predict`). `sys.argv` could be used to simulate the command line call.