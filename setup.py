import os
import subprocess
import argparse


def source_environment_and_run_setup(setvars_path):
    # Change directory to where setup.py is located
    setup_dir = os.path.dirname(os.path.abspath(__file__)) + "/cbm"
    os.chdir(setup_dir)

    # Combine source and setup.py execution into one command
    combined_command = f"source {setvars_path} && python setup.py build_ext --inplace"
    subprocess.check_call(combined_command, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Python extension with environment variables sourced from Intel's setvars.sh.")
    parser.add_argument("--setvars_path",
                        type=str, default="/opt/intel/oneapi/setvars.sh",
                        help="The path to the setvars.sh script to source Intel environment variables (default: /opt/intel/oneapi/setvars.sh)")
    args = parser.parse_args()

    source_environment_and_run_setup(args.setvars_path)
