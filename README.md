## Main Results
![Screenshot](result.png)


## Training command example

```
python train.py \
        --epoch 100 \
        --batch_size 16 \
        --num_workers 2 \
        --dropout_rate 0.3 \
        --train_keys [path_to_train_key] \
        --test_keys [path_to_test_key] \
        --ckpt [path_to_checkpoint] \
        --tatic static \
        --lr 1e-4
```