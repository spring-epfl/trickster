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

Train the target model:
```
PYTHONPATH=. python scripts/wfp.py train --help
```

Generate provably epsilon-minimal adversarial examples:
```
PYTHONPATH=. python scripts/wfp.py generate --help
```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md)
