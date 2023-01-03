#!/usr/bin/env python3
"""Verify that the benchmark functions produces the same data among TorchAudio and TorchVision
"""


def _parse_args(test_choices):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tv-backend",
        default="video_reader",
    )
    parser.add_argument(
        "--test",
        required=True,
        choices=test_choices,
        help="The name of the test.")
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    parser.add_argument('rest', nargs='*')
    return parser.parse_args()


def _main():
    from . import verify

    tests = {
        "stream": verify._verify_stream,
        "random": verify._verify_random,
    }

    ns = _parse_args(tests.keys())

    import torchvision

    torchvision.set_video_backend(ns.tv_backend)

    tests[ns.test](ns.rest, ns.plot)


if __name__ == "__main__":
    _main()
