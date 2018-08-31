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
* Tao Wang's [website fingerprinting datasets](https://www.cse.ust.hk/~taow/wf/data/) using the following command:

## Scripts

### Website fingerprinting

Find provably epsilon-minimal adversarial examples:
```
PYTHONPATH=. python scripts/wfp_bare.py \
    --iter-lim 100000
    --max-trace-len 6000
    --num-examples 100
    --epsilon 1
    --output output_dataframe.pkl
```

See `scripts/wfp_bare.py --help` for more details on the parameters.
