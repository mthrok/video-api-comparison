#!/usr/bin/env bash

set -ue

num_threads="$1"

python -c """
import torch
import torchaudio
import torchvision

print('# Torch:       ', torch.__version__)
print('# TorchAudio:  ', torchaudio.__version__)
print('#  FFmpeg versions:')
for k, v in torchaudio.utils.ffmpeg_utils.get_versions().items():
    print(f'#     {k}: {v}')
print('# TorchVision: ', torchvision.__version__)
"""
printf "Benchmark code: %s\n" "$(git rev-parse --short @)"
printf "# nprocs: %s\n" "$(nproc)"
printf "# threads: %s\n" "${num_threads}"

mkdir -p data tmp
backends=()
if [[ $(uname) != "Darwin" ]]; then
    backends+=("video_reader")
fi

inspect () {
    file=$1
    printf "  #frames: "
    ffprobe -hide_banner -loglevel error \
            -select_streams v:0 -count_packets -show_entries stream=nb_read_packets \
            -of csv=print_section=0 "${file}"

    printf "  #keyframes: "
    ffprobe -hide_banner -loglevel error \
            -select_streams v:0 -skip_frame nokey -show_entries frame=pkt_pts_time \
            -of csv=print_section=0 "${file}" | sed '/^$/d' | tr '\n' ','
    printf "\n\n"
}

get_decoder_opt () {
    num_threads=$1
    if [ ${num_threads} == 0 ]; then
        echo "{\"threads\": \"auto\"}"
    else
        echo "{\"threads\": \"${num_threads}\"}"
    fi
}

if [ ! -z "${TEST_STREAM-1}" ]; then

    printf "***********************\n"
    printf "Test streaming\n"
    printf "***********************\n"

    for ext in avi mp4; do
        printf "Testing %s\n" "${ext}"
        files=()
        for duration in 1 3 5 10 30; do
            file="data/test_stream_${duration}.${ext}"
            files+=("${file}")

            if [ ! -f "${file}" ]; then
                # printf "Generating test data %s\n" "${file}"
                ffmpeg -hide_banner \
                       -f lavfi -t "${duration}" -i testsrc \
                       -pix_fmt "yuv420p" "${file}" > /dev/null 2>&1
            fi

            if [ ! -z "${VERIFY_FUNC-}" ] ; then
                printf "Verifying that the test functions produce identical results... \n"
                for file in "${files[@]}"; do
                    printf "%s ... " "${file}"
                    python3 -m comp_src  --test stream --tv-backend="video_reader" -- --data "${file}"
                    printf " OK\n"
                done
                printf "\n"
            fi
        done

        printf "Testing TorchAudio\n" 
        for file in "${files[@]}"; do
            printf "%s\t" "${file}"
            python3 -m timeit \
                    --unit msec \
                    --setup \
"""
from comp_src import ta
""" \
"""
ta.test_stream(\"${file}\", decoder_option=$(get_decoder_opt ${num_threads}))
"""
        done
        printf "\n"

        for backend in "${backends[@]}"; do
            printf "Testing TorchVision (%s)\n" "${backend}"

            for file in "${files[@]}"; do
                printf "%s\t" "${file}"
                python3 -m timeit \
                        --unit msec \
                        --setup \
"""
from comp_src import tv 
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_stream(\"${file}\", num_threads=${num_threads})
"""
            done
            printf "\n"
        done
    done
fi # TEST_STREAM

if [ ! -z "${TEST_SEEK-1}" ]; then

    printf "***********************\n"
    printf "Test seek\n"
    printf "***********************\n"

    keyint=250
    duration=30
    for ext in "avi" "mp4"; do
        file="data/test_seek_${keyint}.${ext}"

        if [ ! -f "${file}" ]; then
            # printf "Generating test data %s\n" "${file}"
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

        printf "Testing on %s\n" "${file}"
        inspect "${file}"

        timestamps=(0 5 9 10 15 19 20 25 29)
        if [ ! -z "${VERIFY_FUNC-}" ] ; then
            printf "Verifying that the test functions produce identical results... \n"
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
            printf "\n"
        fi

        printf "Testing TorchAudio\n"
        for ts in "${timestamps[@]}"; do
            printf -v ts "%s," ${ts}
            printf "%s (%s)\t" "${file}" ${ts}

            if [[ $file =~ .*\.avi$ ]]; then
                thread_type="frame"
            else
                thread_type="slice"
            fi

            python3 -m timeit \
                    --unit msec \
                    --setup \
"""
from comp_src import ta
""" \
"""
ta.test_random(\"${file}\", ${ts} decoder_option=$(get_decoder_opt ${num_threads}))
"""
        done
        printf "\n"

        for backend in "${backends[@]}"; do
            printf "Teseting TorchVision (%s)\n" "${backend}"

            for ts in "${timestamps[@]}"; do
                printf -v ts "%s," ${ts}
                printf "%s (%s)\t" "${file}" "${ts}"
                python3 -m timeit \
                        --unit msec \
                        --setup \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_random(\"${file}\", ${ts} num_threads=${num_threads})
"""
            done
            printf "\n"
        done
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
    for file in "${files[@]}"; do
        if [ ! -f "${file}" ]; then
            printf "Fetching test data %s\n" "${file}"
            wget --quiet "https://github.com/pytorch/vision/raw/main/test/assets/videos/$(basename "${file}")" -O "${file}"
        fi
        printf "Testing on %s\n" "${file}"
        inspect "${file}"
    done

    if [ ! -z "${VERIFY_FUNC-}" ] ; then
        printf "Verifying that the test functions produce identical results... \n"
        failed=0
        set +e
        for file in "${files[@]}"; do
            printf "%s ... " "${file}"
            python3 -m comp_src \
                    --test stream \
                    --tv-backend="video_reader" \
                    -- \
                    --data "${file}"
            if [ $? -ne 0 ]; then
                failed=1
                printf "The test returns different results. "
                printf "To plot the difference, please use the following command;\n"
                printf "python3 -m comp_src --test stream --tv-backend=\"video_reader\" --plot -- --data ${file}"
            fi
            printf "\n"
        done
        if [ "$failed" -ne 0 ] ; then
            exit 1
        fi
    fi

    printf "Testing TorchAudio\n"
    for file in "${files[@]}"; do
        printf "%s \t" "${file}"
        python3 -m timeit \
                --unit msec \
                --setup \
"""
from comp_src import ta
""" \
"""
ta.test_stream(\"${file}\", decoder_option=$(get_decoder_opt ${num_threads}))
"""
    done
    printf "\n"

    for backend in "${backends[@]}"; do
        printf "Teseting TorchVision (%s)\n" "${backend}"

        for file in "${files[@]}"; do
            printf "%s \t" "${file}"
            python3 -m timeit \
                    --unit msec \
                    --setup \
"""
from comp_src import tv
import torchvision
torchvision.set_video_backend(\"${backend}\")
""" \
"""
tv.test_stream(\"${file}\", num_threads=${num_threads})
"""
        done
        printf "\n"
    done
fi
