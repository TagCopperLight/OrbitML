#!/usr/bin/env fish
# Builds the Aurora viewer, starts it, then streams a scene of orbitml.ipynb to it.
# Usage: viz/run.fish <scene> [--tle N]   (scenes: 1 3 8 9 13 14 fin, see stream/scenes.py)

set root (status dirname)

cmake -S $root -B $root/build -DCMAKE_BUILD_TYPE=Release >/dev/null; or exit 1
cmake --build $root/build --target orbit_viewer -j4; or exit 1

# Shaders and assets are loaded relative to the working directory
pushd $root/build
./viewer/orbit_viewer &
set viewer_pid $last_pid
popd

sleep 1
uv run --project $root/.. --group viz $root/stream/scenes.py $argv

kill $viewer_pid 2>/dev/null
