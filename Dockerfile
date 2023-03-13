FROM nvcr.io/nvidia/tensorrt:22.12-py3

RUN mkdir -p /controlnet_scribble
ADD infer /controlnet_scribble
WORKDIR /controlnet_scribble

RUN pip3 install -r requirements.txt -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ \
    --trusted-host mirrors.cloud.aliyuncs.com \
    --extra-index-url https://pypi.ngc.nvidia.com

EXPOSE 8080

ENTRYPOINT ["bash", "start.sh"]