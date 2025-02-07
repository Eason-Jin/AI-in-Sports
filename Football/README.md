# AI in Sports
## To Run the Code
### Data Processing
1. The initial dataset is `match_data.csv`. This dataset is very large so I selected the successful actions and saved them to `success.csv`. 
2. `success.csv` contains thousands of matches. Thus each of them is given its own folder under `matches/`. To perform this operation, call the `matchData()` function in `process_data.py`.
3. To prepare the data for PCMCI, GAT, and MLP, call the respective functions in `process_data.py`. You will see `match_data_causal.csv`, `match_data_gat.csv`, and `match_data_mlp.csv` respectively under the desired match subfolder.
### Running PCMCI
1. Run PCMCI on the desired folder, if no parameters are given, it will run on all matches.
2. PCMCI will be performed on each player, generating a link file and a causal graph per player. These results can be aggregated by calling the `aggregateLinks()` function in `pcmci.py`. If no parameters are given, it will search in all match folders and aggregate results. Or you can specify a specific match folder to aggregate (and ignoring others).
### Running GAT
1.Make sure you have generated the `match_data_gat.csv` file for the desired match folder. If not, go back to step 3 of the Data Processing section.
2. Run `gat.py` to train the GAT model. 