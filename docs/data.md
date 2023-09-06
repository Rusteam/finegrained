# Data module

The data module is intended for preparing data for training. 
It heavily depends on the [FiftyOne](https://voxel51.com/docs/fiftyone/index.html) package
and its integrations. 
There are the following submodules:

## Submodules

1. **annotations** - send data for annotation in CVAT and fetch annotations.
2. **brain** - commands for `fiftyone.brain` module.
3. **display** - print dataset-related stats to a console.
4. **export** - export datasets to different formats (if missing in the original fiftyone cli).
5. **tag** - tag dataset samples that meet certain criteria.
6. **transforms** - perform changes on datasets.
7. **zoo** - perform operations with `fiftyone.zoo` module.

::: finegrained.data.annotations
    options:
        show_root_heading: true

::: finegrained.data.brain
    options:
        show_root_heading: true

::: finegrained.data.display
    options:
        show_root_heading: true

::: finegrained.data.export
    options:
        show_root_heading: true

::: finegrained.data.tag
    options:
        show_root_heading: true

::: finegrained.data.transforms
    options:
        show_root_heading: true

::: finegrained.data.zoo
    options:
        show_root_heading: true
