# Inference entrypoint
# - standalone executable script, loading arguments from command line
#   providing convenience for a quick demo

import argparse


def parse_args():
    parser = argparse.ArgumentParser("Inference FastWAM")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pass