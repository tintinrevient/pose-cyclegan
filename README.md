# Pose CycleGAN

## Dataset

Download the chosen dataset:
* apple2orange
* horse2zebra (buggy: some images don't have the RGB channel)
```bash
./download_dataset apple2orange
```

```
├── datasets                   
|   ├── <dataset_name>         # i.e. apple2orange
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e., Apple)
|   |   |   └── B              # Contains domain B images (i.e., Orange)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e., Apple)
|   |   |   └── B              # Contains domain B images (i.e., Orange)
```

## Training

```bash
python train.py --dataset apple2orange --cuda --n_epochs 20 --decay_epoch 10
```

<p float="left">
    <img src="pix/training.png" width="600" />
</p>

## Testing

```bash
python test.py --dataset apple2orange --cuda
```

## Generate

```bash
python generate.py --input_img pix/surf.jpg --generator A2B
python generate.py --input_img pix/nude.jpg --generator B2A
```

<p float="left">
    <img src="pix/surf.jpg" height="256" />
    <img src="pix/surf_output.png" height="256" />
</p>

<p float="left">
    <img src="pix/nude.jpg" height="256" />
    <img src="pix/nude_output.png" height="256" />
</p>

## Losses

<p float="left">
    <img src="pix/loss.png" height="400" />
</p>

## References

* https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
* https://stackoverflow.com/questions/53970733/i-want-to-compute-the-distance-between-two-numpy-histogram