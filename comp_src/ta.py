from torchaudio.io import StreamReader


def test_stream(src, frames_per_chunk=1):
    reader = StreamReader(src)
    reader.add_basic_video_stream(frames_per_chunk=frames_per_chunk)
    chunks = [chunk for chunk, in reader.stream()]
    return chunks
