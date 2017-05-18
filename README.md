# tsne-shots
t-SNE visualizations of players based on shots in the 2015-2016 NBA season.
To get proper representations of players and their likeliness to other players,
we use a Variational Autoencoder neural network (VAE) to get encoded formations
of shots, to compare with other players for clustering in t-SNE.

### Setup
- Get dev dependancies
```
sudo apt-get install python-dev libblas-dev liblapack-dev libatlas-base-dev
```

1. Install python libraries
```
pip install -r requirements.txt
```

2. Collect data
```
python get_data.py
```

3. Plot shot charts
```
python savvas_plots.py
```

### VAE Encoding
1. Move train and test images into folders
```
python create_train_test.py
```

2. Train vae from image folders.
```
python main.py
```


### t-SNE Visualization
