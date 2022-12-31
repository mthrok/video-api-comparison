#!/usr/bin/env python3
"""Verify that the benchmark functions produces the same data among TorchAudio and TorchVision
"""


def _parse_args(test_choices):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the video data used for verification")
    parser.add_argument(
        "--test",
        required=True,
        choices=test_choices,
        help="The name of the test.")
    parser.add_argument(
        "--tv-backend",
        default="video_reader",
    )
    return parser.parse_args()


def _verify_stream(data):
    import torch
    from . import ta
    from . import tv

    ta_chunks = ta.test_stream(data)
    tv_chunks = tv.test_stream(data)

    assert len(ta_chunks) == len(tv_chunks)

    for (ta_chunk, tv_chunk) in zip(ta_chunks, tv_chunks):
        ta_frame = ta_chunk[0].permute((1, 2, 0))
        tv_frame = tv_chunk["data"].permute((1, 2, 0))
        try:
            torch.testing.assert_close(ta_frame, tv_frame)
        except AssertionError:
            plot_frames(ta_frame, tv_frame)
            raise


def _main():
    tests = {
        "stream": _verify_stream
    }

    args = _parse_args(tests.keys())

    import torchvision

    torchvision.set_video_backend(args.tv_backend)

    tests[args.test](args.data)


def plot_frames(ta_frame, tv_frame):
    import torchvision
    import matplotlib.pyplot as plt

    diff = ta_frame - tv_frame

    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True)
    for j in range(3):
        axes[j][0].imshow(ta_frame[..., j])
        axes[j][1].imshow(tv_frame[..., j])
        axes[j][2].imshow(diff[..., j])
    axes[-1][0].imshow(ta_frame)
    axes[-1][1].imshow(tv_frame)
    axes[-1][2].imshow(diff)
    axes[0][0].set_title("TorchAudio")
    axes[0][1].set_title(f"TorchVideo: {torchvision.get_video_backend()}")
    axes[0][2].set_title(f"Diff: {diff.abs().sum()}")
    axes[0][0].set_ylabel("Red")
    axes[1][0].set_ylabel("Green")
    axes[2][0].set_ylabel("Blue")
    axes[-1][0].set_ylabel("RGB")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    _main()
