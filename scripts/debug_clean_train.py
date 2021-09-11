
import sys
from main import main

task = "bi_sst"
config = 'embedding--random__lstm'

config_file = f"experiments/{task}/{config}.json"
serialization_dir = f"models/{task}/{config}"

sys.argv = [
    "python",  # useless since we directly call `main()``
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
]

main()



