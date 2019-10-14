#/bin/bash
/usr/local/cuda-9.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.12
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /home/zhutian/.conda/envs/tf/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -L/home/zhutian/.conda/envs/tf/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 # -D_GLIBCXX_USE_CXX11_ABI=0
