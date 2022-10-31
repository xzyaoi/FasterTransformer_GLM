From nvcr.io/nvidia/pytorch:22.09-py3

RUN conda install -y pybind11
RUN pip3 install icetk transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN mkdir /FasterTransformer
ADD . /FasterTransformer
RUN bash /FasterTransformer/docker/install.sh
CMD ["bash", "/FasterTransformer/docker/start.sh"]
EXPOSE 5000