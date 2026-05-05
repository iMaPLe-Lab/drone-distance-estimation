
# DroneDAR: Long-Range Drone Distance Estimation Using Monocular Vision and Bounding-Box Features

<p align="center" width="100%">
  <img src='https://github.com/iMaPLe-Lab/drone-distance-estimation/blob/main/assets/drone_range_model.png' width="90%">
</p>

This repository contains the code implementation of our 2026 paper, "DroneDAR: Long-Range Drone Distance Estimation Using Monocular Vision and Bounding-Box Features". Our work focuses on the problem of predicting the metric distance from a camera to a drone using only visual inputs. To solve this problem, we present DroneDAR (**Drone** **D**etection **A**nd **R**anging), a novel model for estimating the distance of drones that employs a custom bounding-box feature gate to improve robustness across distance regimes. We perform controlled experiments analyzing the impact of backbone capacity, loss function selection, and crop resolution on range estimation performance, and we report failure modes that arise at long distances.

For more information, please check out our paper: (coming soon)
<!-- <p align="left">
<a href="TBD" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-TBD.TBD-b31b1b.svg?style=flat" /></a>
</p> -->

## Dataset

The LRDDv3 dataset used in this work is available on request through our lab website: [LRDDv3](https://research.coe.drexel.edu/ece/imaple/lrddv3/)

## Repo layout
```text
src/
  dataset.py          # dataset loading code
  model.py            # our model in DistancePredictor class, baseline model in DroneRanger class

drone_range.ipynb     # train and evaluate models. Uses functions from evalute.py and train.py
evaluate.py           # code for running batch inference on test data
train.py              # code for training models
```

## Installation

1. Clone this repo and install requirements:
  
   ```
    git clone https://github.com/iMaPLe-Lab/drone-distance-estimation.git
    cd drone-distance-estimation
    conda env create -f environment.yml
    conda activate DroneRange
   ```
  
2. Set the paths and hyperparameters in:
   `drone_range.ipynb`

   Make sure these are correct for your machine:
   - `data_root`
   - `metadata_dir`

3. Train and evaluate models by selecting parameters in
   ```
   drone_range.ipynb
   ```

## Citation

If you find our paper useful in your work, please consider citing our paper:

```
@inproceedings{peterson2026dronedar,
  title={DroneDAR: Long-Range Drone Distance Estimation Using Monocular Vision and Bounding-Box Features},
  author={Peterson, Knut and Mayers, Zaid and Han, David},
  booktitle={???},
  year={2026}
}
```

## Contact

Feel free to contact us through email if you have any questions.

- Knut Peterson (kp3275@drexel.edu), Drexel University

<p align="right">(<a href="#top">back to top</a>)</p>
