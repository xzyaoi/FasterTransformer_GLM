if ! curl -sL --fail https://google.com --connect-timeout 1 -o /dev/null; then
    echo "Using China Mirror"
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
    conda config --set show_channel_urls yes
fi