import os


def _verify(ta_chunks, tv_chunks, tmp_path):
    import torch

    assert len(ta_chunks) > 0
    assert len(tv_chunks) > 0
    assert len(ta_chunks) == len(tv_chunks)

    print(f"Decoded {sum(len(c) for c in ta_chunks)} frames. ({len(ta_chunks)} chunks) ", end="")

    success = True
    for i, (ta_chunk, tv_chunk) in enumerate(zip(ta_chunks, tv_chunks)):
        try:
            torch.testing.assert_close(ta_chunk, tv_chunk)
        except AssertionError:
            os.makedirs(tmp_path, exist_ok=True)
            for t in range(ta_chunk.size(0)):
                ta_frame = ta_chunk[t].permute((1, 2, 0))
                tv_frame = tv_chunk[t].permute((1, 2, 0))
                plot_frames(ta_frame, tv_frame, f"{tmp_path}/chunk_{i}_frame_{t}.png")
            success = False
    if not success:
        raise RuntimeError(
            f"The data do not match. Checkout the comparison report in {tmp_path}")


def _slugify(string):
    return string.replace("/", "_").replace(" ", "_")


def _verify_stream(args):

    def _parse_args(args):
        import argparse

        parser = argparse.ArgumentParser(
            description="Verify streaming test logics produce the same result.")
        parser.add_argument(
            "--data",
            required=True,
            help="Path to the video data used for verification")
        parser.add_argument(
            "--frames-per-chunk",
            type=int,
            default=1)
        return parser.parse_args(args)

    ns = _parse_args(args)

    from . import ta, tv

    ta_chunks = ta.test_stream(ns.data, ns.frames_per_chunk)
    tv_chunks = tv.test_stream(ns.data, ns.frames_per_chunk)

    tmp_path = f"tmp/{_slugify(ns.data)}_{ns.frames_per_chunk}"
    _verify(ta_chunks, tv_chunks, tmp_path)


def _verify_random(args):

    def _parse_args(args):
        import argparse

        parser = argparse.ArgumentParser(
            description="Verify streaming test logics produce the same result.")
        parser.add_argument(
            "--data",
            required=True,
            help="Path to the video data used for verification")
        parser.add_argument(
            "--timestamps",
            nargs="+",
            required=True,
            type=float,
            help="The time stamps to perform seek."
        )
        return parser.parse_args(args)

    ns = _parse_args(args)

    from . import ta, tv

    ta_chunks = ta.test_random(ns.data, *ns.timestamps)
    tv_chunks = tv.test_random(ns.data, *ns.timestamps)

    tmp_path = f"tmp/{_slugify(ns.data)}_{'_'.join(str(ts) for ts in ns.timestamps)}"
    _verify(ta_chunks, tv_chunks, tmp_path)


def plot_frames(ta_frame, tv_frame, path):
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
    axes[0][1].set_title(f"TorchVision: {torchvision.get_video_backend()}")
    axes[0][2].set_title(f"Diff: {diff.abs().sum()}")
    axes[0][0].set_ylabel("Red")
    axes[1][0].set_ylabel("Green")
    axes[2][0].set_ylabel("Blue")
    axes[-1][0].set_ylabel("RGB")
    fig.tight_layout()
    plt.savefig(path)
