# Runs on the BU SCC

The SCC runs the last commit pushed to branch `web27` of this repo; results come back with `scc/scc.sh fetch`.

| File | Where | Purpose |
|---|---|---|
| `setup.sh` | SCC | one-time setup: clone, uv + locked environment, `runs/` → `/projectnb/PROJECT/$USER/runs`, tests |
| `submit.sh` | SCC | `bash scc/submit.sh CONFIG.toml [SLOTS=16] [HOURS=12]`: dry run (plan, time estimate), then `qsub` |
| `job.sh` | SCC | the batch job: `run.py CONFIG --yes --workers $NSLOTS` |
| `scc.sh` | local | `setup`, `sync`, `submit`, `status`, `log`, `fetch` over ssh (`fetch` merges the SCC's `runs/index.csv` lines into the local one) |

## One-time setup

1. **Account.** An SCC account belongs to a project; the project's PI (faculty) adds you at the RCS project management site (bu.edu/tech/support/research). Note the project name.
2. **ssh alias.** Add to `~/.ssh/config` on your machine (`LOGIN` = BU login name):

   ```
   Host scc
       HostName scc1.bu.edu
       User LOGIN
       ControlMaster auto
       ControlPath ~/.ssh/cm-%r@%h:%p
       ControlPersist 12h
       ServerAliveInterval 60
   ```

   ControlMaster keeps one authenticated connection open for 12 h: you enter the password (and Duo, if asked) once with `ssh scc`, and every later `ssh scc` / `scp` / `rsync` reuses it without asking.
3. **Setup.** `scc/scc.sh setup PROJECT` (about 5 minutes; ends with the test results).

## Each run

```
ssh scc                                   # once per 12 h: password (+ Duo); leave it open or exit
scc/scc.sh submit configs/X.toml 16 12    # push, pull on SCC, dry run, qsub (16 slots, 12 h)
scc/scc.sh status                         # qw = waiting, r = running
scc/scc.sh log                            # progress of the newest job
scc/scc.sh fetch                          # copy runs/ back (or: fetch <run folder name>); merges index.csv
```

`run.toml` of each run records host, job id and slots. Uncommitted local changes are not on the SCC: commit before `submit`.

**Sizes.** All slots are used even for a run with few combinations: slots beyond one per combination run each combination's starts, restarts and bootstrap in parallel. `-pe omp` takes up to 28 slots on most nodes (more on some); jobs wait longer for more slots or hours. Default run time limit 12 h. Time per combination scales roughly with grid cells × iterations; take the local dry-run estimate × (8 / slots).
