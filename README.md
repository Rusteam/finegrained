# Fine-grained image recognition

A package and command-line tool to quickly get from raw data to a production model.
The main target is to train an object detector and a image classfier / similarity search
model to detect and recognize fine-grained objects on images and video.

This package is built upon an open-source computer vision data management tool
[FiftyOne](https://docs.voxel51.com/index.html). It also provides an integration
with a data annotation tool [CVAT](https://www.cvat.ai/) and a model-training framework
[Lightning Flash](https://lightning-flash.readthedocs.io/) that are 
used to prepare data and train models.

## Structure

The package is structured as follows:

- **data**: data prep for training
- **models**: train, evaluate and export models
- **services**: interact with external services

## Development installation

This package is managed with [Poetry](https://python-poetry.org/).
To install the package in development mode, run the following command:

```bash
poetry install
pre-commit install
```
