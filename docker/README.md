## build

```bash
bash docker/build.sh
```

## run without ckpt

```bash
docker run -it --rm --gpus all --shm-size=10g -p 5000:5000 -e DATA_TYPE=int4 -e MPSIZE=8 ftglm:latest
```

## run with ckpt

```bash
docker run -it --rm --gpus all --shm-size=10g -p 5000:5000 -v <your path to checkpoints>/49300:/checkpoints -e DATA_TYPE=int4 ftglm:latest
```

## test

```bash
python3 examples/pytorch/glm/glm_server_test.py
```
