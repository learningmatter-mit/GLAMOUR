## GLAMOUR: Graph Learning over Macromolecule Representations
#### Somesh Mohapatra, Joyce An, Rafael Gómez-Bombarelli
#### Department of Materials Science and Engineering, Massachusetts Institute of Technology

The repository and the [Tutorial](https://github.com/learningmatter-mit/GLAMOUR/blob/main/Tutorial.ipynb) accompanies [GLAMOUR: Graph Learning over Macromolecule Representations](https://arxiv.org/abs/2103.02565).<br>

<img src="https://github.com/learningmatter-mit/GLAMOUR/blob/main/overview.svg" width="100%" height="400"><br>

In this work, we developed a graph representation for macromolecules. Leveraging this representation, we developed methods for - <br>
<ul>
<li><b>Similarity Computation:</b> Using chemical similarity between monomers through cheminformatic fingerprints and exact graph edit distances (GED) or graph kernels to compare topologies, it allows for quantification of the chemical and structural similarity of two arbitrary macromolecule topologies. <br>
<li><b>Unsupervised Learning:</b> Dimensionality reduction of the similarity matrices, followed by coloration using the labels shows distinct regions for different classes of macromolecules. <br>
<li><b>Supervised learning:</b> The representation was coupled to supervised GNN models to learn structure-property relationships in glycans and anti-microbial peptides. <br>
<li><b>Attribution:</b> These methods highlight the regions of the macromolecules and the substructures within the monomers that are most responsible for the predicted properties. <br>
</ul>

### Using the codebase
To use the code with an Anaconda environment, follow the installation procedure here - 
```
conda create -n GLAMOUR python=3.6.12
conda activate GLAMOUR
conda install pytorch==1.7.1 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge matplotlib
conda install -c rdkit rdkit==2018.09.3
conda install -c dglteam dgl-cuda10.1
conda install -c dglteam dgllife
conda install captum -c pytorch
conda install -c anaconda scikit-learn==0.23.2
conda install -c anaconda networkx
conda install seaborn
conda install -c conda-forge svglib
conda install -c conda-forge umap-learn
conda install -c conda-forge grakel
```

If you are new to Anaconda, you can install it from [here](https://www.anaconda.com/).

### How to cite
```
@article{GLAMOUR,
  author    = {Somesh Mohapatra and
               Joyce An and
               Rafael G{\'{o}}mez{-}Bombarelli},
  title     = {Chemistry-informed Macromolecule Graph Representation for Similarity
               Computation and Supervised Learning},
  journal   = {CoRR},
  volume    = {abs/2103.02565},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.02565},
  archivePrefix = {arXiv},
  eprint    = {2103.02565},
}
```

### License
MIT License
