import subprocess
import click

@click.command()
@click.option('--minimal_core_count', default=1)
@click.option('--maximal_core_count', default=10)
def main(minimal_core_count, maximal_core_count):
    for i in range(minimal_core_count, maximal_core_count+1):
        subprocess.run(f"python ncf.py --desc weak_scaling_run_{i}_cores_{i}_epochs --threads {i} --data ./data/cache/ml-20m --checkpoint_dir ./data/checkpoints/ --epochs {i}".split())
    

if __name__ == "__main__":
    main()
