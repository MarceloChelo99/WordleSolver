import sys
import subprocess
import os
from pathlib import Path



def install_requirements():
    print(sys.executable)
    subprocess.run([sys.executable, '-m',  'pip', 'install',  '--no-cache-dir', '-r', 'requirements.txt'])


def create_virtualenv():
    subprocess.run(['python', '-m', 'venv', '.venv'])


def check_virtualenv():
    expected_exec_path = Path(__file__).resolve().parent / '.venv' / 'Scripts' / 'python.exe'
    python_exec_path = Path(sys.executable).resolve()

    if expected_exec_path.exists():
        print('- .venv exists')

    while not expected_exec_path.exists():
        print('- .venv does not exist, creating now')
        create_virtualenv()

    if expected_exec_path.exists():
        print('- .venv found')

    same = (expected_exec_path == python_exec_path)
    if same:
        print('- Correct virtual env activated')
    else:
        print('- Wrong virtual env activated or no none activated')
        return False

    return True


if check_virtualenv():
    install_requirements()
