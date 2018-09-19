#!/bin/sh

echo "Running SGS exampel 0"
python examples/policy.py --depth_image data/sgs/depth_0.npy --segmask data/sgs/mask_0.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 1"
python examples/policy.py --depth_image data/sgs/depth_1.npy --segmask data/sgs/mask_1.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 2"
python examples/policy.py --depth_image data/sgs/depth_2.npy --segmask data/sgs/mask_2.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 3"
python examples/policy.py --depth_image data/sgs/depth_3.npy --segmask data/sgs/mask_3.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 4"
python examples/policy.py --depth_image data/sgs/depth_4.npy --segmask data/sgs/mask_4.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 5"
python examples/policy.py --depth_image data/sgs/depth_5.npy --segmask data/sgs/mask_5.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 6"
python examples/policy.py --depth_image data/sgs/depth_6.npy --segmask data/sgs/mask_6.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 7"
python examples/policy.py --depth_image data/sgs/depth_7.npy --segmask data/sgs/mask_7.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 8"
python examples/policy.py --depth_image data/sgs/depth_8.npy --segmask data/sgs/mask_8.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

echo "Running SGS exampel 9"
python examples/policy.py --depth_image data/sgs/depth_9.npy --segmask data/sgs/mask_9.png --config_filename cfg/examples/dex-net_3.0_wenjie.yaml

