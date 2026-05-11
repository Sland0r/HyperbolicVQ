# NLP Experiments

This folder contains the implementation of the NLP experiments for the Hyperbolic Residual Quantization paper.

Specifically, it implements the **Hierarchy Modeling** experiment on the WordNet noun taxonomy.

## Structure
- `wordnet_dataset.py`: Downloads and processes the transitive closure of WordNet hypernymy relationships to form pairs, and samples negative samples.
- `train_hierarchy.py`: Trains the contrastive InfoNCE hierarchy learning using the `HRQModel` mapped into hyperbolic space with the repo's `ResidualVectorQuantizer` module (`c > 0`).

## Running the code
Ensure the environment is activated, and run the training script:
```bash
./run_experiments.sh
```
