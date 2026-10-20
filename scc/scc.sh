#!/bin/bash
# Local helper (on your machine) for the SCC; needs the ssh host alias `scc` (scc/README.md).
#   scc/scc.sh setup PROJECT          copy setup.sh to the SCC and run it
#   scc/scc.sh sync                   push branch web27 to GitHub, pull it on the SCC
#   scc/scc.sh submit CONFIG [SLOTS] [HOURS]   sync, then submit CONFIG as a batch job
#   scc/scc.sh status                 your queued and running jobs
#   scc/scc.sh log [N]                last N lines (default 40) of the newest job log
#   scc/scc.sh fetch [RUN]            copy run folders (or one run) from the SCC to runs/; the
#                                     SCC's runs/index.csv lines are merged into the local one
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=audited-auctions-empirical

# Append to runs/index.csv the SCC's index lines of runs now in runs/ and not yet listed.
merge_index() {
    local remote=runs/.index-scc.csv
    scp -q "scc:$REMOTE/runs/index.csv" "$remote" 2>/dev/null || return 0
    [ -s runs/index.csv ] || head -n 1 "$remote" > runs/index.csv
    awk -F, 'NR == FNR { listed[$1] = 1; next }
             FNR > 1 && !($1 in listed) && system("test -d \"runs/" $1 "\"") == 0' \
        runs/index.csv "$remote" >> runs/index.csv
    rm -f "$remote"
}

case "${1:-}" in
    setup)
        scp scc/setup.sh scc:setup-audited-auctions.sh
        ssh -t scc "bash -l setup-audited-auctions.sh ${2:?PROJECT}" ;;
    sync)
        [ -z "$(git status --porcelain --untracked-files=no)" ] ||
            echo "warning: uncommitted changes stay local; the SCC runs the last pushed commit"
        git push origin web27
        ssh scc "git -C $REMOTE pull --ff-only" ;;
    submit)
        "$0" sync
        ssh scc "bash -lc 'cd $REMOTE && bash scc/submit.sh ${2:?CONFIG} ${3:-16} ${4:-12}'" ;;
    status)
        ssh scc 'bash -lc "qstat -u \$USER"' ;;
    log)
        ssh scc "bash -lc 'tail -n ${2:-40} \$(ls -t $REMOTE/scc/logs/* | head -1)'" ;;
    fetch)
        mkdir -p runs
        if [ -n "${2:-}" ]; then
            rsync -avh "scc:$REMOTE/runs/$2" runs/
        else
            rsync -avh --exclude /index.csv "scc:$REMOTE/runs/" runs/
        fi
        merge_index ;;
    *)
        sed -n '2,9p' "$0"; exit 1 ;;
esac
