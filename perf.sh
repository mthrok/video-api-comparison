#!/usr/bin/env bash

set -ue

printf "TorchVision %s\n" "$(python -c 'import torchvision;print(torchvision.__version__)')"
printf "TorchAudio %s\n" "$(python -c 'import torchaudio;print(torchaudio.__version__)')"
printf "FFmpeg versions:\n%s\n" "$(python -c '''
import torchaudio
for k, v in torchaudio.utils.ffmpeg_utils.get_versions().items():
    print(f"  {k}: {v}")
''')"


mkdir -p data tmp

backends=()
if [[ $(uname) != "Darwin" ]]; then
    backends+=("video_reader")
fi

if [ ! -z "${TEST_STREAM-1}" ]; then

    printf "***********************\n"
    printf "Test streaming\n"
    printf "***********************\n"

    files=()
    for ext in avi mp4; do
        for duration in 1 5 10 30 60 300; do
            file="data/test_stream_${duration}.${ext}"
            files+=("${file}")

            if [ ! -f "${file}" ]; then
                printf "Generating test data %s\n" "${file}"
                ffmpeg -hide_banner -f lavfi -t "${duration}" -i testsrc "${file}" > /dev/null 2>&1
            fi
        done
    done

    if [ ! -z "${VERIFY_FUNC-}" ] ; then
        printf "Verifying that the test functions produce identical results... \n"
        for file in "${files[@]}"; do
            printf "%s... " "${file}"
            python3 -m comp_src  --test stream --tv-backend="video_reader" -- --data "${file}"
            printf "OK\n"
        done
        printf "\n"
    fi

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
    printf "\n"

    for backend in "${backends[@]}"; do
        printf "Testing TorchVision (%s)\n" "${backend}"

        for file in "${files[@]}"; do
            printf "%s\t" "${file}"
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
        printf "\n"
    done
fi # TEST_STREAM

if [ ! -z "${TEST_SEEK-1}" ]; then

    printf "***********************\n"
    printf "Test seek\n"
    printf "***********************\n"

    files=()
    for keyint in 250; do
        for ext in "mp4" "avi"; do
            file="data/test_seek_${keyint}.${ext}"
            files+=("${file}")

            duration=40
            if [ ! -f "${file}" ]; then
                printf "Generating test data %s\n" "${file}"
                if [ "${ext}" == "mp4" ]; then
                    ffmpeg -hide_banner -f lavfi -t "${duration}" -i mptestsrc \
                           -c:v libx264 \
                           -x264opts "keyint=${keyint}:min-keyint=${keyint}:no-scenecut" \
                           "${file}" > /dev/null 2>&1
                else
                    ffmpeg -hide_banner -f lavfi -t "${duration}" -i mptestsrc \
                           -g "${keyint}" \
                           "${file}" > /dev/null 2>&1
                fi
            fi

            printf "Inspecting test data %s\n" "${file}"
            printf "  #frames: "
            ffprobe -hide_banner -loglevel error \
                    -select_streams v:0 -count_packets -show_entries stream=nb_read_packets \
                    -of csv=print_section=0 "${file}"
            printf "  #key frames: \n"
            ffprobe -hide_banner -loglevel error \
                    -select_streams v:0 -skip_frame nokey -show_entries frame=pkt_pts_time \
                    -of csv=print_section=0 "${file}" | sed '/^$/d'
        done
    done

    timestamps=(0 10 20 30 1 11 21 31 3 13 23 33 5 15 25 35 9 19 29 39)
    if [ ! -z "${VERIFY_FUNC-}" ] ; then
        printf "Verifying that the test functions produce identical results... \n"
        for file in "${files[@]}"; do
            for ts in "${timestamps[@]}"; do
                printf -v ts_ "%s," ${ts}
                printf "%s (%s) ... " "${file}" "${ts_}"
                python3 -m comp_src \
                        --test random \
                        --tv-backend="video_reader" \
                        -- \
                        --data "${file}" \
                        --timestamps ${ts}
                printf "OK\n"
            done
        done
        printf "\n"
    fi

    printf "Testing TorchAudio\n"
    for file in "${files[@]}"; do
        for ts in "${timestamps[@]}"; do
            printf -v ts "%s," ${ts}
            printf "%s (%s)\t" "${file}" ${ts}
            python3 -m timeit \
                    --setup \
"""
from comp_src import ta
""" \
"""
ta.test_random(\"${file}\", ${ts})
"""
        done
    done
    printf "\n"

    for backend in "${backends[@]}"; do
        printf "Teseting TorchVision (%s)\n" "${backend}"

        for file in "${files[@]}"; do
            for ts in "${timestamps[@]}"; do
                printf -v ts "%s," ${ts}
                printf "%s (%s)\t" "${file}" "${ts}"
                python3 -m timeit \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_random(\"${file}\", ${ts})
"""
            done
        done
        printf "\n"
    done
fi # TEST_SEEK
