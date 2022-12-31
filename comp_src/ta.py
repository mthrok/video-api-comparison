from torchaudio.io import StreamReader


def test_stream(src, frames_per_chunk=1):
    reader = StreamReader(src)
    reader.add_basic_video_stream(frames_per_chunk=frames_per_chunk)
    chunks = [chunk for chunk, in reader.stream()]
    return chunks



def test_random(src, *timestamps, frames_per_chunk=3):
    reader = StreamReader(src)
    reader.add_basic_video_stream(frames_per_chunk=frames_per_chunk)
    streamer = reader.stream()
    chunks = []
    for ts in timestamps:
        reader.seek(ts, mode="precise")
        chunk, = next(streamer)
        chunks.append(chunk)
    return chunks
