BUILD_DIR := "./build"

# Shows a list of recipes and their helps
help:
    just --list

# Runs cmake setup
setup:
    cmake -S . -B {{BUILD_DIR}} -DCMAKE_BUILD_TYPE=Release

# Builds the project
build:
    cd {{BUILD_DIR}} && make

run: build
    ./build/stream --mode=gpu samples/ACET.mp4 --output=output.mp4

cpu: build
    ./build/stream --mode=cpu samples/ACET.mp4 --output=output.mp4

visualize: run
    mpv output.mp4

