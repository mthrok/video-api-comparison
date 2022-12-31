import torchvision


def test_stream(src):
    reader = torchvision.io.VideoReader(src, "video", num_threads=1)
    chunks = list(reader)
    return chunks
