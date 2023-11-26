# InstaDeep


## Tensorboard

```shell
tensorboard --logdir lightning_logs/
```

-> http://localhost:6006/


```
 python src/main.py --predict --data_path ./tests/data --num_workers 7 --sequence LTDYDNIRNCCKEATVCPKCWKFMVLAVKILDFLLDDMFGFN
```


```
docker build -t proteine_classifier .
docker run -it proteine_classifier
```