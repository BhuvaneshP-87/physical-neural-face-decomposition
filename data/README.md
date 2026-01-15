# Data Layout

Use this directory for face images and any intermediate assets you want to keep local.

Suggested subfolders:

- `data/raw/`: original face images
- `data/masks/`: optional face or skin masks
- `data/aligned/`: pre-aligned crops
- `data/synthetic/`: rendered or procedurally generated samples

The repository ships with a synthetic dataset generator, so you can exercise the full pipeline before wiring in a real face corpus.

