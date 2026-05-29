# Hierarchy Discovery (Section 4.2)
This folder contains the materials to reproduce the Hierarchy Discovery portion of the paper using the Amazon Reviews 2014 dataset.

## Structure
- `train_vae.py`: Trains the RQ-VAE and HRQ-VAE autoencoders to produce multitokens from MPNet continuous embeddings without explicit hierarchical supervision.
- `train_recommender.py`: Trains a sequence-to-sequence recommender system on the generated tokens to predict the next item in a user's purchase history.
- `run_discovery.sh`: SLURM script to execute the pipeline.

## Excluded Content
As requested, any experiments requiring text generation using LLMs (e.g., Claude for the MovieLens dataset) have been completely removed.
