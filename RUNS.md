# Runs behind the paper

Commands are run from the repository root. `jobs/gh_bisect.sh <stem> <args>`
runs `src/vacuum/gh_bisect.py <args> --out <stem>` on one GPU and writes
`<stem>_round<k>.json`, `<stem>_pair.npz` (the bracket's checkpoints) and the
log `<stem>.txt`; `--members` evolves the given amplitudes once and writes
`<stem>_members.npz` instead of a new bracket; `--axis` adds the axis profiles
`<stem>_round<k>_axis.npz`. Every round file records the run's options under
`args`. Archived are the round files, the logs, the axis profiles and the analysis
output; files that hold evolved states are not, and a run that starts from
one (`--init`, `--init-members`, `--dir`, `gh_response.py <pair>`) needs the
step that wrote it first.

    V=python
    A=src/vacuum/gh_analysis.py
    S() { n=$1; shift; sbatch -J $n --gres=gpu:1 -t 14:00:00 --mem=120G jobs/gh_gpu.sh "$@"; }

## Bisection and families (Secs. II, III A, V)

`out/gh/bis_gpu/t24d` is the bisection of the $\eta=1$ family on the
$400\times960$ grid; `e090t`, `e095`/`e095r`, `e105`/`e105r`, `e110t`,
`e120t` and `h085` are the families $\eta=0.9$ to $1.2$ and $0.85$ (`e095r`
and `e105r` continue `e095` and `e105` from their pair files). These were run
with an earlier command line of `gh_bisect.py`, which read the configuration
from a template file; their round files record it under `template`. With the
current command line a family is bisected as

    sbatch -J h085 jobs/gh_bisect.sh out/gh/bis_gpu/h085 --eta 0.85 --lo -3.6215 --hi -3.616 --rounds 4 --set cbad=2.0
    $V $A brackets out/gh/bis_gpu/t24d                     # brackets and A*

Horizons far above the threshold (Sec. II D; `out/gh/ah/ahctl_m*.txt`):

    sbatch -J ahctl -t 12:00:00 jobs/gh_bisect.sh out/gh/bis_gpu/ahctl --lo -4.7 --hi -3.5 --rounds 1 --Tend 40 \
        --members=-4.7,-4.3,-4.0,-3.8,-3.7,-3.6,-3.55,-3.52 --track out/gh/bis_gpu/t24d_round0.json
    sbatch jobs/gh_py.sh out/gh/ah/ahctl_m0.txt $A horizons out/gh/bis_gpu/ahctl_members.npz \
        out/gh/bis_gpu/ahctl_round0.json 0 1,2,3,5,7,9          # one job per member

Ricci terms in $e_c$ along the threshold trajectory (Sec. II E;
`out/gh/thr/weyl_t24d.txt`):

    sbatch jobs/gh_py.sh out/gh/thr/weyl_t24d.txt $A weyl out/gh/bis_gpu/t24d_pair.npz \
        out/gh/bis_gpu/t24d_round0.json 4,8,12,16,20,22,24,26,28,30,32,34,36,38

## Echoes against tuning; gauge and resolution (Sec. III, Figs. 1, 2)

Members at $A_\star\pm\Delta A$ with $A_\star=-3.5091138668$:

    sbatch -J ghwide -t 16:00:00 jobs/gh_bisect.sh out/gh/thr/wide --lo -3.52 --hi -3.49 --rounds 1 --Tend 46 --set ckpt=1.0 \
        --track out/gh/bis_gpu/t24d_round0.json \
        --members=-3.5191138668,-3.5121138668,-3.5101138668,-3.5094138668,-3.5092138668,-3.5091438668,-3.5091238668,-3.5091168668,-3.5091108668,-3.5091038668,-3.5090838668,-3.5090138668,-3.5088138668,-3.5081138668,-3.5061138668,-3.4991138668
    M="--members=-3.5091238668,-3.5091168668,-3.5091148668,-3.5091141668,-3.5091139668,-3.5091138968,-3.5091138768,-3.50911387026,-3.50911386334,-3.5091138568,-3.5091138368,-3.5091137668,-3.5091135668,-3.5091128668,-3.5091108668,-3.5091038668"
    sbatch -J ghg1 -t 24:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/g1 --lo -3.5092 --hi -3.5090 --rounds 1 --Tend 60 $M \
        --set ckpt=10.0 --set hold=40 --set muL=6.0 --set muS=30.0      # doubled damping, zoom not replayed
    sbatch -J ghg2 -t 24:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/g2 --lo -3.5092 --hi -3.5090 --rounds 1 --Tend 60 $M \
        --set ckpt=10.0 --set hold=40 --set muL=1.5 --set muS=7.5       # halved damping
    sbatch -J ghr500 -t 40:00:00 --mem=160G jobs/gh_bisect.sh out/gh/thr/r500 --lo -3.5092 --hi -3.5090 --rounds 1 --Tend 60 $M \
        --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --set ckpt=10.0 --set hold=40 \
        --set n=500 --set nz=1200 --set "regrid=[20.0, 500, 8.0]"
    $V $A departure out/gh/bis_gpu/t24d_round0.json out/gh/thr/wide_round0.json   # Fig. 1(b), Sec. III B
    $V $A departure out/gh/bis_gpu/t24d_round1.json out/gh/thr/g1_round0.json
    $V $A departure out/gh/thr/r500_round0.json

The fourth-echo test without a collapse criterion (Sec. III C):

    S="--lo -3.5092 --hi -3.5090 --rounds 1 --Tend 46 --set ckpt=10.0 --track out/gh/bis_gpu/t24d_round0.json"
    sbatch -J scanc -t 16:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/scanc $S \
        --members=$(python3 -c "print(','.join(f'{-3.5091168618+1.57e-7*i:.13f}' for i in range(16)))")
    sbatch -J scand -t 16:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/scand $S \
        --members=$(python3 -c "print(','.join(f'{-3.5091124968+1.09e-7*i:.13f}' for i in range(16)))")
    $V $A lobe4 out/gh/bis_gpu/t24d_round0.json out/gh/bis_gpu/t24d_round1.json out/gh/thr/scanc_round0.json out/gh/thr/scand_round0.json

Null rays (Fig. 2): the axis and equatorial lines of four members, then
`gh_null.py` on each (`out/gh/null/`):

    K="--lo -3.5092 --hi -3.5090 --rounds 1 --Tend 62 --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --set ckpt=10.0 --set hold=40"
    sbatch -J nlines -t 20:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/nlines $K --set lines=1 \
        --members=-3.50911387026,-3.50911386334,-3.5091128668,-3.5091038668
    for m in 0 1 2 3; do for w in axis equator; do
      sbatch -c 1 --mem=16G jobs/gh_py.sh out/gh/null/m${m}_$w.txt src/vacuum/gh_null.py out/gh/thr/nlines_round0_lines.npz \
          out/gh/thr/nlines_round0.json $m $w 2.06 out/gh/null/m${m}_$w.npz; done; done

## Linear response (Sec. IV, Tables I and II)

The base is the midpoint of the flip pair of `t24d`; $e_c$ is recorded every
16 substeps (`--set efine=16`). Four batches of the ten polynomial directions
give their tuned content, from which `qfree` builds the nine untuned Brill
directions:

    B1="src/vacuum/gh_response.py out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round1.json --T0 0 --track out/gh/bis_gpu/t24d_round0.json --nrand 0 --Tend 50"
    S q1a out/gh/resp/t24d_T0q1a.txt $B1 --nq 12 --qsel 0:6 --seed 11 --eps 3e-9 --out out/gh/resp/t24d_T0q1a
    # t24d_T0q1b: --qsel 6:12; t24d_T0q2a, t24d_T0q2b: the same with --seed 12
    $V $A qfree out/gh/resp/t24d_T0q{1a,1b,2a,2b}_round0.json out/gh/resp/qfree.npz
    B0="src/vacuum/gh_response.py out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round1.json --T0 0 --track out/gh/bis_gpu/t24d_round0.json --qcoef out/gh/resp/qfree.npz --nrand 0 --epsA 3e-9 --Tend 38.5 --set efine=16"
    S qfa4f out/gh/resp/t24d_T0qfa4f.txt $B0 --qsel 0:6 --eps 8e-7 --out out/gh/resp/t24d_T0qfa4f
    S qfb4f out/gh/resp/t24d_T0qfb4f.txt $B0 --qsel 6:9 --dT --eps 8e-7 --out out/gh/resp/t24d_T0qfb4f
    # t24d_T0qf{a,b}f: the same with --eps 2e-7 (linearity); t24d_T0qf{a,b}4f5: --eps 8e-7 with
    #   --set n=500 --set nz=1200 --set "regrid=[20.0,500,8.0]" (resolution)
    $V $A gram out/gh/resp/t24d_T0qf{a,b}f_round0.json out/gh/resp/t24d_T0qf_gram.npz      # input metric
    $V $A growth out/gh/resp/t24d_T0qf{a,b}4f_round0.json --gram out/gh/resp/t24d_T0qf_gram.npz \
        --cmp out/gh/resp/t24d_T0qfaf_round0.json,out/gh/resp/t24d_T0qfbf_round0.json      # Table I, Sec. IV B, C
    #   (add --perwin to remove R_A with each window's own coefficient)
    $V $A combos out/gh/resp/t24d_T0qf{a,b}4f_round0.json --gram out/gh/resp/t24d_T0qf_gram.npz 2 out/gh/thr/combos2.npz
    B2="src/vacuum/gh_response.py out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round1.json --T0 0 --track out/gh/bis_gpu/t24d_round0.json --qcoef out/gh/thr/combos2.npz --nrand 0 --epsA 3e-9 --eps 8e-7 --Tend 38.5 --set efine=16"
    S qc2 out/gh/resp/t24d_T0qc2.txt $B2 --every 4 --out out/gh/resp/t24d_T0qc2     # t24d_T0qc2h4: add --qh 1e-4
    $V $A gram out/gh/resp/t24d_T0qc2_round0.json out/gh/resp/t24d_T0qc2_gram.npz
    $V $A growth out/gh/resp/t24d_T0qc2_round0.json --gram out/gh/resp/t24d_T0qc2_gram.npz \
        --cmp out/gh/resp/t24d_T0qc2h4_round0.json --floor " -2.59,-2.22,-1.80:-1.73,-1.73,-0.74"   # the leading plane
    $V $A weylresp out/gh/resp/t24d_T0qc2_states.npz out/gh/resp/t24d_T0qc2_round0.json        # Ricci terms (thr/weyl_qc2.txt)

Random fields and family differences from $T=8$ (Sec. IV C, Table II):

    B8="src/vacuum/gh_response.py out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round1.json --T0 8 --track out/gh/bis_gpu/t24d_round0.json --eps 1e-8 --Tend 38.5 --set efine=16"
    D3="--dir out/gh/bis_gpu/e105_pair.npz --dir out/gh/bis_gpu/e095_pair.npz --dir out/gh/bis_gpu/h085_pair.npz"
    S T8f out/gh/resp/t24d_T8f.txt $B8 $D3 --nrand 3 --seed 1 --out out/gh/resp/t24d_T8f
    S T8rf out/gh/resp/t24d_T8rf.txt $B8 --nrand 6 --seed 2 --out out/gh/resp/t24d_T8rf
    S T8axf out/gh/resp/t24d_T8axf.txt $B8 $D3 --nrand 3 --seed 3 --axis 8,2 --out out/gh/resp/t24d_T8axf
    S T8dTf out/gh/resp/t24d_T8dTf.txt $B8 --nrand 5 --seed 4 --axis 8,2 --dT --out out/gh/resp/t24d_T8dTf
    S T8lof out/gh/resp/t24d_T8lof.txt $B8 $D3 --nrand 3 --seed 1 --axis 8,2 --set n=320 --set nz=768 \
        --set "regrid=[20, 320, 8.0]" --out out/gh/resp/t24d_T8lof               # 320 x 768
    S e105f out/gh/resp/e105_T8f.txt src/vacuum/gh_response.py out/gh/bis_gpu/e105_pair.npz out/gh/bis_gpu/e105_round4.json --T0 8 \
        --track out/gh/bis_gpu/e105_round0.json,out/gh/bis_gpu/e105_round1.json,out/gh/bis_gpu/e105_round2.json,out/gh/bis_gpu/e105_round3.json \
        --nrand 6 --seed 5 --eps 1e-8 --Tend 44 --axis 8,2 --set efine=16 --out out/gh/resp/e105_T8f
    # e095_T8f: the same from e095_pair.npz and e095_round3.json, tracking rounds 0-2
    $V $A gram out/gh/resp/t24d_T8{f,rf,axf,dTf}_round0.json out/gh/resp/t24d_T8_gram.npz   # e105_T8, e095_T8 likewise
    $V $A growth out/gh/resp/t24d_T8{f,rf,axf,dTf}_round0.json --gram out/gh/resp/t24d_T8_gram.npz
    $V $A growth out/gh/resp/e105_T8f_round0.json --gram out/gh/resp/e105_T8_gram.npz       # and e095
    $V $A growth out/gh/resp/t24d_T8f_round0.json      # against t24d_T8lof
    $V $A axis out/gh/resp/t24d_T8{axf,dTf}_round0.json

The outputs of these `growth` runs are archived as `out/gh/thr/growth_*.txt`.

## Tuning along the untuned directions (Sec. V, Table III, Fig. 3)

The knobs come from two earlier response batches of the untuned directions
that record $e_c$ only at the refreshes. The archived `qfknobs.npz` keeps their coefficients; the
deformations on the grid are rebuilt by `knobs`. At each point $(p_1,p_2)$ the
bracket is centered on the linear prediction plus
$1.54\times10^{-3}p_1^2+5.38\times10^{-3}p_2^2$:

    S qfa out/gh/resp/t24d_T0qfa.txt $B1 --qcoef out/gh/resp/qfree.npz --qsel 0:6 --eps 2e-7 --epsA 3e-9 --out out/gh/resp/t24d_T0qfa
    S qfb out/gh/resp/t24d_T0qfb.txt $B1 --qcoef out/gh/resp/qfree.npz --qsel 6:9 --dT --eps 2e-7 --epsA 3e-9 --out out/gh/resp/t24d_T0qfb
    $V src/vacuum/gh_codim.py knobs out/gh/resp/t24d_T0qf{a,b}_round0.json --out out/gh/codim/qfknobs.npz
    $V src/vacuum/gh_codim.py pair out/gh/codim/qfknobs.npz --p=1,1 --shift 6.92e-3 --width 1.6e-2 --out out/gh/codim/qf+10_+10_init.npz
    sbatch -J q11 jobs/gh_bisect.sh out/gh/codim/qf+10_+10 --init out/gh/codim/qf+10_+10_init.npz \
        --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --rounds 5 --Tend 70 \
        --set hold=40 --set cbad=0.5 --set ckpt=2.0
    $V src/vacuum/gh_codim.py report --ref out/gh/bis_gpu/t24d_round1.json:5,6 out/gh/codim/qf+10_+10_round4.json   # one row

Runs whose names end in `b` or `c` continue a point on a segment of its pair
(`--init <pair or init file> --sub S0,S1`). The $500\times1200$ rechecks
`r500_qf*` evolve eight members near each threshold from initial data built
with `pair ... --grid 500,1200`, with `--rounds 1 --set n=500 --set nz=1200
--set "regrid=[20.0, 500, 8.0]" --members=...`.

## After the echoes (Sec. VI, Figs. 4 to 6)

`rc021` restarts the 16 members of `t24d` round 1 from $T=20$ and saves them
at $T=39.9$; `dhold` continues them with the zoom fixed from $T=40$ (Fig. 5).
`r400`, `r500`, `r600` rebuild the same members on three grids with the
bisection's zoom (`--init out/gh/bis_gpu/t24d_pair.npz --init-T 20`); `dh500`,
`dh600` repeat `dhold`'s bracket on $500\times1200$ and $600\times1440$.

    sbatch -J rc021 jobs/gh_bisect.sh out/gh/bis_res/rc021 --init out/gh/bis_gpu/t24d_pair.npz --init-T 20 \
        --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --rounds 1 --Tend 72 \
        --set ckpt=100 --axis 3,1 --save-T 39.9 --recentre 40,0.021,0.3
    sbatch -J bdhold -t 20:00:00 --mem=120G jobs/gh_bisect.sh out/gh/bis_res/dhold --init-members out/gh/bis_res/rc021_T39.9.npz \
        --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --set ckpt=2.0 --set cbad=0.5 --set hold=40 \
        --rounds 4 --Tend 140 --axis 3,2
    B="--init out/gh/bis_gpu/t24d_pair.npz --init-T 20 --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json"
    sbatch -J bres500 -t 20:00:00 --mem=120G jobs/gh_bisect.sh out/gh/bis_res/r500 $B --set ckpt=2.0 --B 16 --rounds 1 \
        --Tend 72 --axis 8,2 --set nz=1200 --set "regrid=[20.0, 500, 8.0]"     # r400: production grid; r600: 600 x 1440
    D="--init out/gh/bis_res/dhold_pair.npz --init-T 40 --track out/gh/bis_res/dhold_round0.json --set ckpt=2.0 --set hold=40 --axis 3,2"
    sbatch -J bdh600 -t 24:00:00 --mem=220G jobs/gh_bisect.sh out/gh/bis_res/dh600 $D --Tend 100 --set cbad=0.5 --rounds 2 \
        --set nz=1440 --set "regrid=[20.0, 600, 8.0]"                          # dh500: 500 x 1200
    $V $A curvature out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round0.json hi 41,42,42.5,43,43.5,44,44.5,45   # bis_gpu/t24d_curvature_hi.txt

The deciding event on $600\times1440$ inside `dhold`'s bracket: `nb600`
(`--sub` on a segment, `--set cmax=1e9 --set cbad=1e9`, no constraint
criterion), `noc600` (`--set cmax=1e9`, given members), `dg600` (`--set
cbad=3.0`), and `nh12`, which switches to 1200 radial nodes at $T=52$
(`--set "regrid2=[52.0, 1200, 1440]"`):

    $V $A decide -3.5091138656445 out/gh/bis_res/{nb600,noc600,dg600,dh600}_round0.json
    $V $A decide -3.5091138659775 out/gh/bis_res/nh12_round{0,1}.json

Bisection with 1200 and 1800 radial nodes from $T=52$, then 16 members inside
the last bracket with their states at the deciding event:

    R="--init out/gh/bis_gpu/t24d_pair.npz --init-T 20 --set ckpt=2.0 --set hold=40 --set cmax=1e9 --set cbad=1e9 --set dhold=1.5 --rounds 1 --Tend 60 --axis 3,2"
    T="--track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json"
    sbatch -J s18 jobs/gh_bisect.sh out/gh/bis_res/s18 $R $T --sub 0.3550,0.4060 --set "regrid2=[52.0,1800,1440]"
    sbatch -J s12 jobs/gh_bisect.sh out/gh/bis_res/s12 $R $T --sub 0.3550,0.4060 --set "regrid2=[52.0,1200,1440]"
    sbatch -J s18b jobs/gh_bisect.sh out/gh/bis_res/s18b $R $T,out/gh/bis_res/s18_round0.json --sub 0.3850,0.3910 --set "regrid2=[52.0,1800,1440]"
    sbatch -J s12b jobs/gh_bisect.sh out/gh/bis_res/s12b $R $T,out/gh/bis_res/s12_round0.json --sub 0.3850,0.3910 --set "regrid2=[52.0,1200,1440]"
    $V $A decide -3.50911386599308 out/gh/bis_res/s18_round0.json out/gh/bis_res/s18b_round0.json   # s12, s12b likewise
    M=$(python3 -c "lo=-3.5091138662629757; hi=-3.509113865972318; s=(hi-lo)/7; print(','.join(f'{lo+x*s:.16f}' for x in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.25,6.5,6.75,7]))")
    C="--init out/gh/bis_res/s18b_pair.npz --init-T 28 --set ckpt=2.0 --set hold=40 --set cmax=1e9 --set cbad=1e9 --set dhold=1.5 --rounds 1 --Tend 60.1 --axis 3,2 --save-T 53.8 --members=$M"
    sbatch -J rc18 jobs/gh_bisect.sh out/gh/bis_res/rc18 $C $T,out/gh/bis_res/s18_round0.json,out/gh/bis_res/s18b_round0.json --set "regrid2=[52.0,1800,1440]"
    sbatch -J rc12 jobs/gh_bisect.sh out/gh/bis_res/rc12 $C $T,out/gh/bis_res/s12_round0.json,out/gh/bis_res/s12b_round0.json --set "regrid2=[52.0,1200,1440]"
    $V $A ricci out/gh/bis_res/rc18_T53.8.npz out/gh/bis_res/rc18_round0.json 0,3,6,9,12,15        # bis_res/rc18_ricci.txt
    $V $A constraint out/gh/bis_res/rc18_T53.8.npz out/gh/bis_res/rc18_round0.json 0,3,9,15        # bis_res/rc18_constraint.txt
    $V $A axisah out/gh/bis_res/rc18_members.npz out/gh/bis_res/rc18_round0.json 0,2 56,58,60      # ah/rc18_axisah.txt

Largest Kretschmann scalar on the grid (Fig. 6):

    K="--lo -3.5092 --hi -3.5090 --rounds 1 --Tend 62 --track out/gh/bis_gpu/t24d_round0.json,out/gh/bis_gpu/t24d_round1.json --set ckpt=10.0 --set hold=40"
    sbatch -J imaxf -t 20:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/imaxf $K --set kret=1 \
        --members=-3.5091108668,-3.5091038668,-3.5090838668,-3.5090138668,-3.5088138668,-3.5081138668,-3.5061138668,-3.4991138668,-3.4791138668,-3.4491138668,-3.4091138668,-3.3591138668,-3.2591138668,-3.1091138668,-2.9091138668,-2.5091138668
    sbatch -J imaxn -t 20:00:00 --mem=120G jobs/gh_bisect.sh out/gh/thr/imaxn $K --set kret=1 \
        --members=-3.50911386334,-3.5091138628,-3.5091138618,-3.5091138598,-3.5091138568,-3.5091138518,-3.5091138418,-3.5091138268,-3.5091138068,-3.5091137668,-3.5091137168,-3.5091136168,-3.5091134668,-3.5091132668,-3.5091128668,-3.5091118668
    $V $A imax out/gh/thr/imaxf_round0.json out/gh/thr/imaxn_round0.json

`out/gh/thr/bbc23_fig1_Aneg.txt` is the $A<0$ curve of Fig. 1 of Baumgarte
et al., Phys. Rev. Lett. 131, 181401 (2023), read off the figure.

## Linearized evolution map (Sec. VII)

    sbatch jobs/gh_gpu.sh out/gh/linmap/t24d_T11.txt src/vacuum/gh_linmap.py \
        out/gh/bis_gpu/t24d_pair.npz out/gh/bis_gpu/t24d_round0.json --T 11 --out out/gh/linmap/t24d_T11
