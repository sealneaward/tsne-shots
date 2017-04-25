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

- Install python libraries
```
pip install -r requirements.txt
```

- Collect data
```
python get_data.py
```

- Plot shot charts
```
python savvas_plots.py
```
