"""
Utilities for submitting jobs to SLURM system on Discovery cluster.

Example sbatch submission:

    sbatch --nodes=4 --mem=64Gb --time=00:01:00 --job-name=niklas_test \
        --partition=multigpu --gres=gpu:v100-sxm2:2 --output=sbatch_logs.txt \
        --open-mode=append  --wrap="echo STANDARD_OUT_TEST; echo \
        STANDARD_ERR_TEST >&2; echo FILE_CONTENTS_TEST > ~/scrap.txt"

Example interactive submission:

    srun -p multigpu --gres=gpu:v100-sxm2:2 --pty /bin/bash

Example STDOUT from batch submission:
    
    Submitted batch job 9581653

"""
import unittest
import subprocess
import re
from functools import partial


def submit_job(command, partition='short',
        duration_hours='24', duration_minutes='00', job_name='ryanHeminway', mem_gb=32, n_cpu=4,
        logfile='/home/heminway.r/sbatch_logs.txt', openmode='append'):
    command = ' '.join(['sbatch', 
                       f'--nodes={n_cpu}',
                       f'--time={duration_hours}:{duration_minutes}:00',
                       f'--job-name={job_name}',
                       f'--partition={partition}',
                       f'--mem={mem_gb}Gb',
                       f'--ntasks-per-node=1',
                       f'--cpus-per-task=32',
                       f'--output={logfile}',
                       # Note that we single-quote the command for safety
                       f"--wrap='{command}'"])

    err_fn = partial(send_mail, subject='DISCOVERY_JOB_SUBMIT_FAIL',
                  body=f'Failed to submit job with command: "{command}"')

    stdout = run_cmd(command, 'resubmission failed!', err_fn)

    job_id = re.compile('Submitted batch job (\d+)').search(stdout).group(1)

    # send_mail(subject='DISCOVERY_JOB_SUBMIT_SUCCESS',
    #           body=f'Successfully submitted job {job_id}, using command:\n\n{command}')
    return

def send_mail(body, subject, recip='heminway.r@northeastern.edu'):
    # Wrap body in heredoc for safety
    recip1 = 'heminway.r@northeastern.edu'
    command = f"mail -s '{subject}' {recip1} << END_MAIL\n{body}\nEND_MAIL"

    run_cmd(command, 'mailing failed!')
    return

def run_cmd(command, err_msg, err_fn=None):
    if type(command) is list:
        command = ' '.join(command)

    res = subprocess.run(command, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        if err_fn:
            err_fn()
        raise OSError(res.stderr + "\n" + err_msg)

    return res.stdout

class TestSlurm(unittest.TestCase):
    def test_slurm_utils(self):
        run_cmd('echo stdout', 'err_msg')

        with self.assertRaises(OSError):
            run_cmd('exit 1', 'err_msg')

        with self.assertRaises(OSError):
            run_cmd('echo stderr >&2; exit 1', 'err_msg')

        send_mail(subject='TEST_SEND_MAIL',
                  body='TEST_BODY')

        submit_job('echo STDOUT; echo STDERR>&2; echo TEST > /home/smedemark-margulie.n/TestSlurm.output.txt',
                   duration_hours='00', duration_minutes='01', job_name='niklas_test_batch_submit1', 
                   logfile='/home/smedemark-margulie.n/TEST_sbatch_logfile')

        submit_job('/scratch/smedemark-margulie.n/cnaps/src/wrapper.sh python /scratch/smedemark-margulie.n/cnaps/src/test_python_script.py arg1 arg2 arg3',
                   duration_hours='00', duration_minutes='01', job_name='niklas_test_batch_submit2', 
                   logfile='/home/smedemark-margulie.n/TEST_sbatch_logfile')


if __name__ == '__main__':
    unittest.main()
