```
    parameter_filename=args.param_path
    serialization_dir=args.serialization_dir
    overrides=args.overrides
    node_rank=args.node_rank,
    include_package=args.include_package,
    dry_run=args.dry_run,
    file_friendly_logging=args.file_friendly_logging,
        
    
    params = Params.from_file(parameter_filename, overrides)


    training_util.create_serialization_dir(params, serialization_dir, recover=False, force=False)
    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    model: Optional[Model] = None

    distributed_params = params.params.pop("distributed", None)
    # If distributed isn't in the config and the config contains strictly
    # one cuda device, we just run a single training process.
    if distributed_params is None:
        train_loop = TrainModel.from_params(
        params=params,
        serialization_dir=serialization_dir,
        local_rank=0,
        )
        metrics = train_loop.run()
        train_loop.finish(metrics)
        model = train_loop.model
        
    # Otherwise, we are running multiple processes for training.
    else:
        raise Exception('distributed training is not supported')

    archive_model(serialization_dir, include_in_archive=include_in_archive)
```
