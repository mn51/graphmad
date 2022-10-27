# Graph Mixup for Augmenting Data (GraphMAD): Graph Mixup for Data Augmentation using Data-Driven Convex Clustering.

This code implements Graph Mixup for Augmenting Data (GraphMAD), a data-driven nonlinear mixup method for generating new labeled samples from existing labeled graph data to improve graph classification performance.
Mixup of graph data is challenging due to the irregular and non-Euclidean nature of graphs.
Hence, we project the graphs onto a common latent feature space and explore linear and nonlinear mixup strategies in this latent space.
In this work (i) we present nonlinear graph mixup in an interpretable continuous domain given by graphons, random graph models that can represent families of graphs sharing similar structural characteristics, (ii) we apply convex clustering via robust CARP [2,3] to efficiently and accurately learn data-driven mixup functions, where generated samples exploit relationships among all graphs as opposed to pairs of data, and (iii) we compare applying different mixup functions for data samples and their labels. Our work submitted to ICASSP 2023 explores datasets for which this is beneficial.

If you use this code in your research, please cite our paper.

[1] M. Navarro and S. Segarra. "GraphMAD: Graph Mixup for Data Augmentation using Data-Driven Convex Clustering." Submitted to ICASSP 2023.<br>
[2] M. Weylandt, J. Nagorski, G. I. Allen. "Dynamic Visualization and Fast Computation for Convex Clustering via Algorithmic Regularization", Journal of Computational and Graphical Statistics, 29.1, 87-96, 2020.<br>
[3] Q. Wang, P. Gong, S. Chang, T. S. Huang, and J. Zhou. "Robust Convex Clustering Analysis." ICDM, 1263-1268, 2016.
