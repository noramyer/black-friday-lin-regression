# A look at Black Friday purchases using linear regression
### By Nora Myer, Matthew Frey, Haseeb Javed, Aidan Globus

## Quick Start
The following dependencies are required:
- numpy
- scipy
- scikit-learn
- pandas
- matplotlib

To ensure you get an older version of matlabplot that is compatible with python 2.7.X, use:
```
$ python -mpip install -U matplotlib
```

#### Running model selections
To fit the data to a specific model, run:
```
$ python src/model.py --dt 2500
```
or try running multiple models at once:

```
$ python src/model.py --los --ridge .1 --dt 2500
```

Param selection options:
```
--dt 2500     #decision tree, takes 1 arg which is samples/split
--los         #linear model with least ordinary squares, no args
--ridge .1    #ridge regression, takes 1 arg which is alpha
--lasso .1    #lasso linear model, takes 1 arg which is alpha
--forest      #rnd forest regressor, no args
--all         #run all models with basic args
--ablation    #run ablation test specified models
```
#### Running exploratory script
The file exploratory.py takes a look at the raw data before it is processed. It looks at the types of features, uniqueness of features, and purchase averages based on categories. The purpose is to get a baseline purchase estimation and give us a better understading of what the data looks like to help us in the pre-processing stage.

```
$ python src/exploratory.py
```
