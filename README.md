# tsne-shots
t-SNE visualizations of players based on shots in the 2015-2016 NBA season

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

## t-SNE

Using t-SNE embeddings of high dimensional data, low dimensional representations can be made for clustering.
To create clusters and the visualizations of the clusters, execute this command.
```
python t-sne.py
```
