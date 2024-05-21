import subprocess
import os
import sys

def create_conda_env(env_name, python_version):
    """
    Create a conda environment with the specified name and Python version.

    Parameters:
    env_name (str): The name of the conda environment to create.
    python_version (str): The Python version to use for the new environment.

    Raises:
    subprocess.CalledProcessError: If the environment creation fails.
    
    """
    try:
        # Check if the environment already exists
        envs = subprocess.check_output(['conda', 'env', 'list']).decode('utf-8')
        if env_name in envs:
            print(f"Environment '{env_name}' already exists.")
        else:
            # Create the environment with the specified Python version
            subprocess.check_call(['conda', 'create', '--name', env_name, f'python={python_version}', '-y'])
            print(f"Environment '{env_name}' with Python {python_version} created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating the environment: {e}")
        sys.exit(1)

def activate_conda_env(env_name):
    """
    Generate the command to activate a conda environment.

    Parameters:
    env_name (str): The name of the conda environment to activate.

    Returns:
    str: The command to activate the specified conda environment.
    """
    if os.name == 'nt':  # Windows
        activate_cmd = f'conda activate {env_name} && '
    else:  # Unix-based systems
        activate_cmd = f'source activate {env_name} && '
    return activate_cmd

def install_requirements(activate_cmd):
    """
    Install the packages listed in requirements.txt using pip in the activated conda environment.

    Parameters:
    activate_cmd (str): The command to activate the conda environment.

    Raises:
    subprocess.CalledProcessError: If the package installation fails.
    """
    try:
        # Install packages from requirements.txt
        subprocess.check_call(activate_cmd + f'pip install -r requirements.txt', shell=True)
        print("Packages from requirements.txt installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while installing packages: {e}")
        sys.exit(1)

if __name__ == "__main__":
    env_name = "giumeh" 
    python_version = "3.11.9"

    create_conda_env(env_name, python_version)
    activate_cmd = activate_conda_env(env_name)
    install_requirements(activate_cmd)
