# trickster

Experiments with adversarial examples in discrete domains.

## Setup

### Python packages
Install the required Python packages:

```
pip install -r requirements.txt
```

### System packages
On Ubuntu, you need these:
```
apt install parallel unzip
```

### Data

To download the datasets, run this:

```
make data
```

The datasets include:
* Tao Wang's [website fingerprinting](https://www.cse.ust.hk/~taow/wf/data/) "knndata"
* Zafar Gilani's [Twitter bot classification dataset](https://www.cl.cam.ac.uk/~szuhg2/data.html)

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
