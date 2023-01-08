# ReforesTree-Remastered

ReforesTree is designed to be a benchmark dataset of forest carbon stock that would encourage scalable financing schemes
to protect the forests. To avoid the expensive and subjective manually labelled trees, the authors leverage the recent
advancements of AI and propose a fully automatic data processing pipeline. Furthermore, a baseline CNN is implemented to
showcase its superiority over models trained on satellite data when forecasting the carbon stock on small-scale,
tropical agroforestry sites. This project will thoroughly present their data, go through every step in the pipeline,
reproduce their experiments with the baseline CNN, and compare the expected vs actual results.

Link to original repo: https://github.com/gyrrei/ReforesTree

Sample of annotated images:

![](images/annotated_data/3.png)

# Tree detection with deepforest

To reproduce tree detection with deepforest, an older version of deepforest must be installed. 
This is done by cloning the repo and installing deepforest from source.
To avoid issues with tensorflow versions, we recommend developing inside a Docker container.
We included a Dockerfile and a `devcontainer.json` file.

Before building the devcontainer, run the following bash commands:

```bash
export REFORESTREE_WORKSPACE=/path/to/workspace
git clone git@github.com:weecology/DeepForest.git --branch v0.3.2 $REFORESTREE_WORKSPACE
```

