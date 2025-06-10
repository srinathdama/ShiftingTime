## Instructions

The following instructions for dataset preprocessing are obtained from [KNF github](https://github.com/google-research/google-research/tree/master/KNF).

### Dataset and Preprocessing
- M4: Download and preprocess M4 data
```
python M4_KNF/m4_data_gen.py
```

- Cryptos: Download `train.csv` and `asset_details.csv` from [kaggle](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data) to current `data/Cryptos` folder. Run `cryptos_data_gen.py` to preprocess Cryptos data
```
python Cryptos/cryptos_data_gen.py
```

- Traj: Download [NBA basketball player trajectory data](https://github.com/linouk23/NBA-Player-Movements/tree/master/data) and unzip all .7z files in `PlayerTraj/json_data`. Run `traj_data_gen.py` to preprocess Trajectory data. Since we didn't fix random seed when we sampled trajectory, to reproduce the results, please download [the same KNF traj dataset we used](https://drive.google.com/drive/folders/1N_wo1I7G62HglyML5yL4FTEzbfekh4vZ?usp=sharing).
```
python PlayerTraj/traj_data_gen.py
```