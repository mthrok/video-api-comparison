#!/usr/bin/env bash

set -ue

mkdir -p data

files=()
for ext in avi mp4; do
    for duration in 1 5 10 30 60; do
        file="data/test_${duration}.${ext}"
        files+=("${file}")

        if [ ! -f "${file}" ]; then
            printf "Generating test data %s\n" "${file}"
            ffmpeg -hide_banner -f lavfi -t "${duration}" -i testsrc "${file}" > /dev/null 2>&1
        fi
    done
done

backends=("pyav")
if [[ $(uname) != "Darwin" ]]; then
    backends+=("video_reader")
fi

printf "***********************\n"
printf "Test stream\n"
printf "***********************\n"

printf "Verifying that the test functions produce identical results... "
for file in "${files[@]}"; do
    python -m comp_src  --test stream --data "${file}"
done
printf "OK\n"

printf "Testing TorchAudio\n"
for file in "${files[@]}"; do
    printf "%s\t" "${file}"

    python3 -m timeit \
            --setup \
"""
from comp_src import ta
""" \
"""
ta.test_stream(\"${file}\")
"""
done

for backend in "${backends[@]}"; do
    printf "Testing TorchVision - %s\n" "${backend}"
    for file in "${files[@]}"; do
        printf "TorchVision (pyav)\t%s\t" "${file}"
        python3 -m timeit \
                --setup \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_stream(\"${file}\")
"""
    done
done
