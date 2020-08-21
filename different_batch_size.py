import subprocess
import click

@click.command()
@click.option('--minimal_batch_size', default=17)
@click.option('--maximal_batch_size', default=23)
def main(minimal_batch_size, maximal_batch_size):
    for i in range(minimal_batch_size, maximal_batch_size+1):
        subprocess.run(f"python ncf.py --valid_batch_size {2**i} --batch_size {2**i} --desc different_batch --threads 10 --data /data/cache/ml-20m --checkpoint_dir /data/checkpoints/ --epochs 1".split())


if __name__ == "__main__":
    main()
