import os
from slurm_util import submit_job

# Search over mutation ranges
for mutation in [0.01, 0.0098, 0.012]:
    # Search over crossover ranges
    for crossover in [0.5, 1.0]:
            run_name = 'minFeatsGaussian'
            command = f'/bin/bash launch.sh {mutation} {crossover}'
            job_name = f'{run_name}.mut={mutation}.cross={crossover}'
            submit_job(
                command=command,
                partition='short',
                duration_hours='24',
                job_name=job_name,
                mem_gb=24,
                n_cpu=1,
                logfile=f'/home/heminway.r/logs/{job_name}.LOG')

 

