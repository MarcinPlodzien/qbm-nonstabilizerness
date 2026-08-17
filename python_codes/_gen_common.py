"""Shared launcher for the get_data_figure_*.py scripts: run a data-generation
script as a subprocess with a single-threaded CPU environment."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, **env):
    e = dict(os.environ, OMP_NUM_THREADS="1", JAX_PLATFORMS="cpu",
             XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    e.update({k: str(v) for k, v in env.items()})
    print(f">>> {script}  {env}", flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script)], env=e, check=True)
