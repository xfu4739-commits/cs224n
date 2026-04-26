欢迎来到作业 2！

本次作业会使用 PyTorch。如果你对 PyTorch 不熟悉，或想复习 PyTorch 的基础内容，可以在 Canvas 的 Course Videos 下找到 PyTorch 回顾课程视频。

如果你想继续使用作业 1 的 `cs224n` 环境，请确认已经安装 `local_env.yml` 中列出的所有依赖。可以按下面步骤执行：

# 1. 激活旧环境：

    conda activate cs224n

# 2. 安装 docopt

    conda install docopt

# 3. 安装 pytorch、torchvision 和 tqdm

    conda install pytorch torchvision -c pytorch
    conda install -c anaconda tqdm


如果你想为本次作业新建一个环境，请执行：

# 1. 按 local_env.yml 创建环境（根据电脑配置，可能需要一段时间）：
    
    conda env create -f local_env.yml

# 2. 激活新环境：
    
    conda activate cs224n_a2
    

# 如需退出当前激活的环境，使用：
    
    conda deactivate

## Windows 下没有 `zip` 命令
如果你在 Windows 上运行 `collect_submission.sh` 脚本时出现报错，通常是因为 Windows 默认没有 `zip` 命令。你可以参考 [这里](https://superuser.com/questions/201371/create-zip-folder-from-the-command-line-windows) 提到的其他打包方式；或者直接手动把 Python 文件压缩成 zip 提交。