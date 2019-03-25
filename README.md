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
To fit the data to different modes
```
$ python src/model.py --dt 2500
```
or try

```
$ python src/model.py --los --dt 2500
```

Param selection options:
```
--dt 2500   #decision tree, takes 1 arg which is samples/split
--los   #linear model with least ordinary squares, no args
```

## About the data

## About the code
