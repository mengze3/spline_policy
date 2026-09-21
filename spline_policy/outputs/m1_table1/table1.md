| Method | 3 | 7 | 12 | 17 | 22 | Closest-point query |
| --- | --- | --- | --- | --- | --- | --- |
| Piecewise | 7.97 ± 2.76 | 3.40 ± 1.10 | 1.97 ± 0.63 | 1.39 ± 0.44 | 1.08 ± 0.35 | Discrete lookup |
| B.P. | 6.18 ± 3.65 | 0.96 ± 0.60 | 0.28 ± 0.13 | 0.14 ± 0.06 | 0.10 ± 0.04 | Polynomial |
| RBF | 16.51 ± 4.35 | 1.79 ± 0.61 | 0.26 ± 0.08 | 0.12 ± 0.04 | 0.05 ± 0.02 | Numerical |
| Fourier / DCT (FAST-cont.) | 5.18 ± 3.52 | 0.56 ± 0.33 | 0.19 ± 0.08 | 0.09 ± 0.04 | 0.05 ± 0.02 | Numerical |
| Q.S. (ours) | 6.55 ± 3.12 | 0.74 ± 0.42 | 0.21 ± 0.09 | 0.11 ± 0.05 | 0.06 ± 0.02 | Cubic / segment |

Reconstruction error on 210 LASA demonstrations (1000 original samples each). Entries are the mean and population standard deviation of per-trajectory mean pointwise Euclidean errors in original LASA coordinate units; lower is better. K is the number of independent real coefficients per coordinate (2K scalars in 2D). Q.S. uses K-1 C1-continuous quadratic segments with zero terminal velocity, retaining K free coefficients per coordinate after eliminating one coefficient. Fourier/DCT is grouped as one cosine-basis family; the row reports FAST's continuous orthonormal DCT with K retained low frequencies, without quantization or BPE. The RCFS mirrored-Fourier fit is retained in the raw results. Closest-point query describes the construction, not a measured runtime: B.P. has a stationarity polynomial of degree 2K-3; Q.S. requires at most cubic roots per segment, with endpoints also checked. Smooth alternative bases also admit numerical flow-field construction.
