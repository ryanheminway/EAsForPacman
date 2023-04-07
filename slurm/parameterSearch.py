import os
from slurm_util import submit_job

# Search over mutation ranges
for mutation in [0.1, 0.05]:
    # Search over crossover ranges
    for crossover in [0, 1.0, 0.8, 0.6]:
        for selection in ["roulette", "truncated", "tournament"]:
            for runIdx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                run_name = 'selCompare'
                command = f'/bin/bash launch.sh {mutation} {crossover} {selection} {runIdx}'
                job_name = f'{run_name}.mut={mutation}.cross={crossover}.sel={selection}.run={runIdx}'
                submit_job(
                    command=command,
                    partition='short',
                    duration_hours='24',
                    job_name=job_name,
                    mem_gb=24,
                    n_cpu=1,
                    logfile=f'/home/heminway.r/logs/{job_name}.LOG')

            



 

