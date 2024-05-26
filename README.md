# Category_Aware Active Domain Adaptation
Pytorch implementation for paper **Category-Aware Active Domain Adaptation**. The full code will be released shortly.

# Abstract
Active domain adaptation has shown promising results in enhancing unsupervised domain adaptation (DA), by actively selecting and annotating a small amount of unlabeled samples from the target domain. Despite its effectiveness in boosting overall performance, the gain usually concentrates on the categories that are readily improvable, while challenging categories that demand the utmost attention are often overlooked by existing models. To alleviate this discrepancy, we propose a novel category-aware active DA method that aims to boost the adaptation for the individual category without adversely affecting others. Specifically, our approach identifies the unlabeled data that are most important for the recognition of the targeted category. Our method assesses the impact of each unlabeled sample on the recognition loss of the target data via the influence function, which allows us to directly evaluate the sample importance, without relying on indirect measurements used by existing methods. Comprehensive experiments and in-depth explorations demonstrate the efficacy of our method on category-aware active DA over three datasets.

# Framework
![Alt text](framework.png?raw=true "Title")


# Prerequisites
- python >= 3.6.8
- pytorch ==>=1.7.0
- torchvision == >=0.5.0
- numpy, scipy, PIL, argparse, tqdm, pandas,prettytable,scikit-learn,webcolors,matplotlib,opencv-python,numba

# Base DA methods
We run our base DA methods based on the implementation of [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library).
We use the default setting the in their [example codes](https://github.com/thuml/Transfer-Learning-Library/tree/master/examples/domain_adaptation/image_classification) to run DANN.




# Acknowledgement
This project is built on the open-source [CLUE](https://github.com/virajprabhu/CLUE) and [influence function](https://github.com/brandeis-machine-learning/influence-fairness) implementations. Thank the authors for their excellent work.
