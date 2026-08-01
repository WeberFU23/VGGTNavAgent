#!/bin/bash
# 创建 VGGT-SLAM 运行环境（python 3.11，独立于 habitat py3.9 环境）。
# 用法: bash scripts/setup_vggtslam.sh
set -e
# 兼容不同机器的 conda 安装位置
for _conda_sh in ~/anaconda3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$_conda_sh" ]; then source "$_conda_sh"; break; fi
done

conda create -n vggtslam python=3.11 -y
conda activate vggtslam

cd "$(dirname "$0")/../VGGT-SLAM"

pip install -r requirements.txt

mkdir -p third_party
cd third_party
if [ ! -d salad ]; then
  git clone https://github.com/Dominic101/salad.git
fi
pip install -e ./salad
if [ ! -d vggt ]; then
  git clone https://github.com/MIT-SPARK/VGGT_SPARK.git vggt
fi
pip install -e ./vggt
cd ..

pip install -e .

python -c "import gtsam; assert hasattr(gtsam, 'SL4'), 'gtsam SL4 missing'; print('gtsam SL4 ok')"
echo "SETUP_DONE"
