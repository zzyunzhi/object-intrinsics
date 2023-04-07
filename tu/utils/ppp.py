import subprocess
from tu.loggers.timer import print_time


def run_command(command, name=None, quiet=False, **kwargs):
    stdout = subprocess.DEVNULL if quiet else None
    result = None
    if name is None:
        try:
            result = subprocess.run(command, check=True, stdout=stdout, **kwargs)
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
        return result

    with print_time(name):
        try:
            result = subprocess.run(command, check=True, stdout=stdout, **kwargs)
        except Exception as e:
            print(e)
        return result