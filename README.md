# Python package for fast graph cut algorithms
The goal of this package is to include a number of fast graph cut algorithms. Currently, the package package only includes our serial and parallel implementations of the [Quadratic Pseudo-Boolean Optimization (QPBO) algorithm](https://github.com/Skielex/QPBO). We plan to include more algorithms in the future.

## Installation
Install from repository (requires `Cython`):
```
clone https://github.com/Skielex/shrdr
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
- [slgbuilder](https://github.com/Skielex/slgbuilder) (CVPR 2020)
- [thinqpbo](https://github.com/Skielex/thinqpbo)
- [thinmaxflow](https://github.com/Skielex/thinmaxflow)

## License
MIT License (see LICENSE file).

## Reference
If you use this for academic work, please consider citing our paper, [Faster Multi-Object Segmentation Using Parallel Quadratic Pseudo-Boolean Optimization](https://openaccess.thecvf.com/content/ICCV2021/html/Jeppesen_Faster_Multi-Object_Segmentation_Using_Parallel_Quadratic_Pseudo-Boolean_Optimization_ICCV_2021_paper.html).