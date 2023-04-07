import os
import torch


def print_slurm_info():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    slurm_job_name = os.environ.get('SLURM_JOB_NAME')
    if slurm_job_id is not None:
        sbatch_output = f"/viscam/u/yzzhang/projects/img/logs/{slurm_job_id}-{slurm_job_name}.out"
        print('running in slurm, output file: ')
        print(sbatch_output)

    print('visible devices', os.environ.get('CUDA_VISIBLE_DEVICES'))

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
