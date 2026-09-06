#!/usr/bin/env bash
# 等主批跑结束后补跑 ep1 替补（4ok3usBNeis_ep0000010）。
# 注意：pgrep 模式用 [.] 避免匹配本脚本自身的命令行。
while pgrep -f 'run_formal10_v28[.]sh' > /dev/null; do
  sleep 60
done
bash /root/autodl-tmp/run_formal10_v28_ep1fix.sh
