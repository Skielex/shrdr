# Python package for fast graph cut algorithms
The goal of this package is to include a number of fast graph cut algorithms. Currently, the package package only includes our serial and parallel implementations of the [Quadratic Pseudo-Boolean Optimization (QPBO) algorithm](https://github.com/Skielex/QPBO). We plan to include more algorithms in the future.

## Installation
Install from repository (requires `Cython`):
```
git clone https://github.com/Skielex/shrdr
cd shrdr
pip install .
```
Package will be made available on PyPI later.

## What is it for?
The package is targeted anyone working with boolean optimization problems, in particular for computer vision. Examples include:
- Minimization of both submodular and non-submodular boolean optimization problems.
- Image segmentation
- Surface detection
- Image restoration
- Image synthesis
- ...and many other tasks.

### Examples
See notebooks available at [doi.org/10.5281/zenodo.5201619](https://doi.org/10.5281/zenodo.5201619) for benchmark examples. More usage examples will come later!

## ICCV 2021 poster video
The M-QPBO and P-QPBO algorithms in the package are being presented at ICCV 2021. We show that P-QPBO is up to 20x faster than the original QPBO algorithm for large segmentation tasks.

[![Teaser video](https://img.youtube.com/vi/79vvYSLXA4s/0.jpg)](https://youtu.be/79vvYSLXA4s)

## Related repositories
- [slgbuilder](https://github.com/Skielex/slgbuilder) Python package (CVPR 2020)
- [thinqpbo](https://github.com/Skielex/thinqpbo) Python package
- [thinmaxflow](https://github.com/Skielex/thinmaxflow) Python package
- [thinhpf](https://github.com/Skielex/thinhpf) Python package
- [C++ implementations](https://github.com/patmjen/maxflow_algorithms) of max-flow/min-cut algorithms

## License
MIT License (see LICENSE file).

## Reference
If you use this any of this for academic work, please consider citing our work:
- [Sparse Layered Graphs for Multi-Object Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Jeppesen_Sparse_Layered_Graphs_for_Multi-Object_Segmentation_CVPR_2020_paper.pdf) 

- [Faster Multi-Object Segmentation using Parallel Quadratic Pseudo-Boolean Optimization](https://openaccess.thecvf.com/content/ICCV2021/papers/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.pdf)



### BibTeX

``` bibtex
@INPROCEEDINGS{9156301,  author={Jeppesen, Niels and Christensen, Anders N. and Dahl, Vedrana A. and Dahl, Anders B.},  booktitle={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   title={Sparse Layered Graphs for Multi-Object Segmentation},   year={2020},  volume={},  number={},  pages={12774-12782},  doi={10.1109/CVPR42600.2020.01279}}

@INPROCEEDINGS{9710633,  author={Jeppesen, Niels and Jensen, Patrick M. and Christensen, Anders N. and Dahl, Anders B. and Dahl, Vedrana A.},  booktitle={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},   title={Faster Multi-Object Segmentation using Parallel Quadratic Pseudo-Boolean Optimization},   year={2021},  volume={},  number={},  pages={6240-6249},  doi={10.1109/ICCV48922.2021.00620}}
```