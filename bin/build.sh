#!/bin/sh
function pkg_deploy() {
    tar_src=`find dist -type f | grep "tar.gz" | grep -v cli`
    pip install -q oss2 -i https://mirrors.aliyun.com/pypi/simple
    python $ROOT/bin/osscli $tar_src alps/`basename $tar_src`
}
cd ..
rm -rf dist
python setup.py sdist

ROOT=$(pwd)
echo $ROOT

pkg_deploy
