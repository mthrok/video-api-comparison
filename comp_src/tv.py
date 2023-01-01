import itertools

import torch
import torchvision


def test_stream(src, frames_per_chunk=1):
    reader = torchvision.io.VideoReader(src, "video", num_threads=1)
    chunks = []
    partial = []
    for c in reader:
        partial.append(c["data"])
        if len(partial) == frames_per_chunk:
            chunks.append(torch.stack(partial, 0))
            partial = []
    if partial:
        chunks.append(torch.stack(partial, 0))
    return chunks


def test_random(src, *timestamps, frames_per_chunk=3):
    reader = torchvision.io.VideoReader(src, "video", num_threads=1)
    chunks = []
    for ts in timestamps:
        reader = reader.seek(ts, keyframes_only=False)
        frames = [c["data"] for c in itertools.islice(reader, frames_per_chunk)]
        chunks.append(torch.stack(frames, 0))
    return chunks
