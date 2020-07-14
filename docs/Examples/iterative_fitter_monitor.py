import matplotlib.pyplot as plt
import pandas as pd

go = 'y'
parameters = 'slope,intercept,beta0,beta1,Cphi'.split(',')
while go == 'y':
    go = input('continue? ')
    plt.close('all')
    df = pd.read_csv('outputs/iterative_fitter_output_df.csv')
    info_cols = {parameter: [] for parameter in parameters}
    for col in df.columns:
        for parameter in parameters:
            if parameter in col:
                info_cols[parameter].append(col)
    for parameter in parameters:
        fig, ax = plt.subplots()
        ax.title

