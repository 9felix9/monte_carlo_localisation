#!/usr/bin/env bash
set -euo pipefail

WS_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$WS_DIR"
colcon build --symlink-install

mkdir -p "$WS_DIR/runtime_logs"

# fake_robot (läuft 3 Sekunden)
gnome-terminal --title="fake_robot" -- zsh -lc "
  cd '$WS_DIR'
  source install/setup.zsh
  echo 'Running fake_robot for 3 seconds...'
  ros2 launch fake_robot fake_robot.launch.py \
    2>&1 | tee '$WS_DIR/runtime_logs/fake_robot.log'
" &

sleep 0.5

# mcl_node (läuft 3 Sekunden)
gnome-terminal --title="mcl_node" -- zsh -lc "
  cd '$WS_DIR'
  source install/setup.zsh
  echo 'Running MCL node for 3 seconds...'
  ros2 run mcl_localization mcl_node --ros-args --log-level INFO \
    2>&1 | tee '$WS_DIR/runtime_logs/mcl_node.log'
" &
