SM=$(cat /FasterTransformer/SM_NUMBER)
mkdir /FasterTransformer/build
cd /FasterTransformer/build && cmake -DSM=$SM -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
cd /FasterTransformer/build && make -j
# cd /FasterTransformer/build && ./bin/gpt_gemm 1 1 128 96 128 49152 150528 1 8 