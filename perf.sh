#!/usr/bin/env bash

set -ue

python -c """
import torch
import torchaudio
import torchvision

print('Torch:       ', torch.__version__)
print('TorchAudio:  ', torchaudio.__version__)
print('  FFmpeg versions:')
for k, v in torchaudio.utils.ffmpeg_utils.get_versions().items():
    print(f'    {k}: {v}')
print('TorchVision: ', torchvision.__version__)
"""

mkdir -p data tmp

backends=()
if [[ $(uname) != "Darwin" ]]; then
    backends+=("video_reader")
fi

if [ ! -z "${TEST_STREAM-1}" ]; then

    printf "***********************\n"
    printf "Test streaming\n"
    printf "***********************\n"

    for ext in avi mp4; do
        printf "Testing %s\n" "${ext}"
        files=()
        for duration in 1 3 6 30 60 300; do
            file="data/test_stream_${duration}.${ext}"
            files+=("${file}")

            if [ ! -f "${file}" ]; then
                printf "Generating test data %s\n" "${file}"
                ffmpeg -hide_banner -f lavfi -t "${duration}" -i testsrc "${file}" > /dev/null 2>&1
            fi
        done

        if [ ! -z "${VERIFY_FUNC-}" ] ; then
            printf "Verifying that the test functions produce identical results... \n"
            for file in "${files[@]}"; do
                for fpc in 1 3 10; do
                    printf "%s (%s)... " "${file}" "${fpc}"
                    python3 -m comp_src  --test stream --tv-backend="video_reader" -- --data "${file}" --frames-per-chunk "${fpc}"
                    printf " OK\n"
                done
            done
            printf "\n"
        fi

        printf "Testing TorchAudio\n" 
        for fpc in 1 3 10; do
            for file in "${files[@]}"; do
                printf "%s (fpc: %s)\t" "${file}" "${fpc}"
                
                python3 -m timeit \
                        --unit msec \
                        --setup \
"""
from comp_src import ta 
ta.test_stream(\"${file}\", ${fpc})  # warming up
""" \
"""
ta.test_stream(\"${file}\", ${fpc})
"""
            done
        done
        printf "\n"

        for backend in "${backends[@]}"; do
            printf "Testing TorchVision (%s)\n" "${backend}"

            for fpc in 1 3 10; do
                for file in "${files[@]}"; do
                    printf "%s (fpc: %s)\t" "${file}" "${fpc}"
                    python3 -m timeit \
                            --unit msec \
                            --setup \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_stream(\"${file}\", ${fpc})
"""
                done
            done
            printf "\n"
        done
    done
fi # TEST_STREAM

if [ ! -z "${TEST_SEEK-1}" ]; then

    printf "***********************\n"
    printf "Test seek\n"
    printf "***********************\n"

    files=()
    for keyint in 250; do
        for ext in "avi" "mp4"; do
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

    timestamps=(0 1 5 9 10 11 15 19 20 21 25 29 30 31 35 39)
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
                    --unit msec \
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
                        --unit msec \
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

if [ ! -z "${TEST_RANDOM_ACCESS-1}" ]; then

    printf "***********************\n"
    printf "Test random access\n"
    printf "***********************\n"

    # The following files have invalid PTS values
    files=(
        "data/v_SoccerJuggling_g23_c01.avi"
        "data/v_SoccerJuggling_g24_c01.avi"
        "data/R6llTwEh07w.mp4"
        "data/SOX5yA1l24A.mp4"
        "data/WUzgd7C1pWA.mp4"
        "data/RATRACE_wave_f_nm_np1_fr_goo_37.avi"
        "data/SchoolRulesHowTheyHelpUs_wave_f_nm_np1_ba_med_0.avi"
        "data/TrumanShow_wave_f_nm_np1_fr_med_26.avi"
    )
    timestamps=(
        '2 4 6'
        '2 4 6'
        '2 4 6'
        '2 4 6'
        '2 4 6'
        '1 2'
        '1 0'
        '1 0'
    )
    for file in "${files[@]}"; do
        if [ ! -f "${file}" ]; then
            printf "Fetching test data %s\n" "${file}"
            wget --quiet "https://github.com/pytorch/vision/raw/main/test/assets/videos/$(basename "${file}")" -O "${file}"
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
        printf "\n"
    done

    if [ ! -z "${VERIFY_FUNC-}" ] ; then
        printf "Verifying that the test functions produce identical results... \n"
        failed=0
        for i in $(seq $((${#files[@]} - 1))); do
            file="${files[i]}"
            ts="${timestamps[i]}"
            printf -v ts_ "%s," ${ts}
            printf "%s (%s) ... \n" "${file}" "${ts_}"
            python3 -m comp_src \
                    --test random \
                    --tv-backend="video_reader" \
                    -- \
                    --data "${file}" \
                    --timestamps ${ts} || failed=1
        done
        if [ "$failed" -ne 0 ] ; then
            exit 1
        fi
        printf "\n"
    fi

    printf "Testing TorchAudio\n"
    for i in $(seq $((${#files[@]} - 1))); do
        file="${files[i]}"
        printf -v ts "%s," ${timestamps[i]}
        printf "%s (%s)\t" "${file}" ${ts}
        python3 -m timeit \
                --unit msec \
                --setup \
"""
from comp_src import ta
""" \
"""
ta.test_random(\"${file}\", ${ts})
"""
    done
    printf "\n"

    for backend in "${backends[@]}"; do
        printf "Teseting TorchVision (%s)\n" "${backend}"

        for i in $(seq $(( ${#files[@]} - 1 ))); do
            file="${files[i]}"
            printf -v ts "%s," ${timestamps[i]}
            printf "%s (%s)\t" "${file}" "${ts}"
            python3 -m timeit \
                    --unit msec \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_random(\"${file}\", ${ts})
"""
        done
        printf "\n"
    done
fi
