import itertools

import torch
import torchvision


def test_stream(src, num_threads=1):
    reader = torchvision.io.VideoReader(src, "video", num_threads=num_threads)
    return [c['data'] for c in reader]


def test_random(src, *timestamps, frames_per_chunk=1, num_threads=1):
    reader = torchvision.io.VideoReader(src, "video", num_threads=num_threads)
    chunks = []
    for ts in timestamps:
        reader = reader.seek(ts, keyframes_only=False)
        frames = [c["data"] for c in itertools.islice(reader, frames_per_chunk)]
        chunks.append(torch.stack(frames, 0))
    return chunks
