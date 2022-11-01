## build

```bash
bash docker/build.sh
```

## run

```bash
docker run -it --rm --gpus all -p 5000:5000 -v <your path to checkpoints>/49300:/checkpoints -v DATA_TYPE=int4 ftglm:latest
```

## test

```bash
python3 examples/pytorch/glm/glm_server_test.py
```
