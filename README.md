# trickster

Experiments with adversarial examples in discrete domains.

## Setup

### Python packages
Install the required Python packages:

```
pip install -r requirements.txt
```

### Data

To download the datasets, run this:

```
make data
```

The datasets include:
* Tao Wang's [website fingerprinting](https://www.cse.ust.hk/~taow/wf/data/) "knndata"

## Scripts

### Website fingerprinting

Find provably epsilon-minimal adversarial examples:
```
PYTHONPATH=. python scripts/wfp.py \
    --iter_lim 100000
    --num_examples 100
    --max_trace_len 6000
    --epsilon 1
    --features cumul
    --dummies_per_insertion 1
    --output output_dataframe.pkl
```

Use the `--help` option for more details about the parameters.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md)
