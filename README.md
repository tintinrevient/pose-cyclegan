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

## Testing

```bash
python test.py --dataset apple2orange --cuda
```

