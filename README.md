# Finer tuning stops adding echoes to centered Brill waves near the black-hole threshold

Evolution code, analysis scripts and archived output for

> Xulei Sun, *Finer tuning stops adding echoes to centered Brill waves near
> the black-hole threshold*.

The vacuum Einstein equations are evolved in generalized-harmonic form in
coordinates that zoom in on the center: the physical metric is
$g=L(T)^2\hat g$ with $d\ln L/dT=-\ell$, and the outer faces of the single
fixed grid are spacelike, so they need no boundary data. Batches of 16
axisymmetric Brill waves are evolved in lockstep on one GPU. A bisection round
restarts from the last common checkpoint of its bracket and replays the zoom
and substeps of the earlier rounds, so the depth is not limited by the
precision of the amplitude and a member's evolution does not depend on its
batch. The paper bisects the centered negative-amplitude Brill family to
$|A-A_\star|=3.5\times10^{-9}$, counts echoes against the tuning, measures the
linear response along the threshold trajectory, bisects again with growing
untuned deformations added to the data, and follows the trajectory after the
echoes.

## Layout

    src/vacuum/   the evolver and its analysis (below)
    jobs/         Slurm wrappers: gh_bisect.sh (bisection, one GPU), gh_gpu.sh, gh_py.sh
    out/gh/       archived output: run records, axis profiles, analysis output
    paper/        make_figs.py, which writes paper/figs/
    RUNS.md       the runs behind each section of the paper, and the commands

| module | |
|---|---|
| `gh_sim.py` | GH evolution in zooming coordinates: CPU reference, grid, gauge, maximal-lapse target, curvature; self-tests |
| `gh_kernels.py` | numba kernels of the pointwise GH algebra |
| `gh_cuda.py` | the same evolution for a batch of states on one GPU (numba.cuda, torch) |
| `gh_data.py` | Brill initial data (Hamiltonian-constraint solve) |
| `gh_ah.py` | apparent horizons, including a search in stretched coordinates next to the axis |
| `gh_bisect.py` | the production configuration `CONFIG` and the threshold bisection |
| `gh_response.py` | linear response along a threshold trajectory, many directions in one batch |
| `gh_codim.py` | deformations along the untuned directions and their initial data; the end-of-echoing criterion |
| `gh_analysis.py` | post-processing of bisections and responses |
| `gh_null.py` | outgoing null geodesics on the axis and in the equatorial plane |
| `gh_linmap.py` | leading eigenvalues of the linearized evolution map over a short time |

Run everything from the repository root; the scripts find their own modules.

## Reproducing the figures

    python paper/make_figs.py          # paper/figs/*.pdf (Figs. 1-6), about 15 s

This reads only `out/gh/` and needs numpy, scipy and matplotlib. Rerun from
this archive, the figures are those of the paper up to the PDF creation date.

## Numbers in the text

The numbers in the text and tables are printed by `gh_analysis.py` and
`gh_codim.py` from the archived run records, without new evolutions, for
example

    python src/vacuum/gh_analysis.py brackets out/gh/bis_gpu/t24d          # A*
    python src/vacuum/gh_analysis.py lobe4 out/gh/bis_gpu/t24d_round0.json out/gh/bis_gpu/t24d_round1.json \
        out/gh/thr/scanc_round0.json out/gh/thr/scand_round0.json          # Sec. III C
    python src/vacuum/gh_analysis.py growth out/gh/resp/t24d_T0qfa4f_round0.json out/gh/resp/t24d_T0qfb4f_round0.json \
        --gram out/gh/resp/t24d_T0qf_gram.npz \
        --cmp out/gh/resp/t24d_T0qfaf_round0.json,out/gh/resp/t24d_T0qfbf_round0.json   # Table I

`RUNS.md` gives the command for each section. Analyses that need the evolved
states (Ricci and constraint residuals at the deciding event, the horizon
searches, the curvature of the first off-center event, the null rays, the
linearized map) are archived as their output.

## Re-running the evolutions

Every run writes `<stem>_round<k>.json`: its options under `args`, the full
configuration under `template`, the bracket, the members with their outcomes,
the zoom rate and substeps of every step, and the history of each member at
every refresh (central Weyl scalar and proper time, extrema of the curvature
and lapse and their positions, constraint norm). `RUNS.md` lists the commands.
Runs that continue an earlier one start from its evolved states
(`<stem>_pair.npz`, `<stem>_members.npz`, `<stem>_T<T>.npz`, several GB
each), which are not archived; rerun the earlier step first.

A bisection round of 16 members on the $400\times960$ production grid takes
3.5 to 5.5 hours on one NVIDIA B200, and a round from $T=0$ keeps every
member's checkpoints in host memory (240 GB). `gh_cuda.py` compiles its
kernels with numba.cuda, which needs libNVVM: set `CUDA_HOME` to a CUDA 13
toolkit (a directory with `nvvm/` and `lib64/`; the libraries of the
`nvidia-*` wheels that come with torch serve). The job scripts take the
interpreter from `PYTHON` and must be submitted from the repository root.
Self-tests:

    python src/vacuum/gh_sim.py        # flat-space RHS, kernels against the reference, Ricci
    python src/vacuum/gh_data.py       # Brill solver
    python src/vacuum/gh_ah.py         # horizon finder
    sbatch jobs/gh_gpu.sh out/gh_cuda_check.txt src/vacuum/gh_cuda.py   # GPU against CPU

## Requirements

Python 3.11 with the versions the results were produced under:

    numpy 2.3.5   scipy 1.17.1   matplotlib 3.10.6   numba 0.65.0   sympy 1.14.0   torch 2.13.0

`pip install -r requirements.txt`. The figures and the analyses of archived
runs need only numpy, scipy and matplotlib; sympy is used by a self-test, and
torch and a GPU only by the evolutions.

## Citation

If you use this code or the archived output, please cite the paper above.
`CITATION.cff` carries the same information in machine-readable form.

## License

Code under the MIT License (`LICENSE`); the archived output in `out/` under
CC BY 4.0.
