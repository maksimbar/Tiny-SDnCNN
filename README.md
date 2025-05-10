

## Generating dataset
Dataset with 60/20/20 splits is already included, you just need to extract it.

However, if you want to generate one from scratch, you can use command:

```
uv run scripts/splitter/main.py data --seed 8000
```

## Running experiments
This projects uses Hydra package to handle different parameter configurations.

### Sanity check
To make sure everything works as expected, run the command bellow. It sets pretty small computational requirements and must work just fine even with a CPU. 

```
uv run train.py \
    data.base_dir='${hydra:runtime.cwd}/data_small' \
    data.sample_rate=8000 \
    train.epochs=5 \
    train.sgd.lr_decay_epochs=5 \
    stft.window_type=hamming \
    model.activation=relu
```

### Window Functions

```
uv run train.py stft.window_type=hamming
```

```
uv run train.py stft.window_type=hann
```

```
uv run train.py stft.window_type=blackman
```

### Activation Functions

```
uv run train.py model.activation=relu
```

```
uv run train.py model.activation=prelu
```

```
uv run train.py model.activation=leaky_relu
```