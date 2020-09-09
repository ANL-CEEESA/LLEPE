import matplotlib.pyplot as plt
import pandas as pd

go = 'y'
parameters = 'slope,intercept,beta0,beta1'.split(',')
# while go == 'y':
#     go = input('continue? ')
#     if go != 'y':
#         break
#     plt.close('all')
df = pd.read_csv('outputs/iterative_fitter_output4.csv')
info_cols = {parameter: [] for parameter in parameters}
for col in df.columns:
    for parameter in parameters:
        if parameter in col:
            info_cols[parameter].append(col)
for parameter in parameters:
    mini_df = df[info_cols[parameter]]
    fig, ax = plt.subplots()
    ax.set_title(parameter)
    for col in info_cols[parameter]:
        ax.plot(df['iter'].values[1:],
                df[col].values[1:],
                label=col,
                linestyle='-',
                marker='o')
    ax.set_xlabel('iteration')
    ax.set_ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/{0}.png'.format(parameter))
fig, ax = plt.subplots()
ax.set_title('best_obj_value')
ax.plot(df['iter'].values[1:],
        df['best_obj'].values[1:],
        linestyle='-',
        marker='o')
ax.set_xlabel('iteration')
ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig('outputs/best_obj.png')
fig, ax = plt.subplots()
ax.set_title('rel_diff')
ax.plot(df['iter'].values[1:],
        df['rel_diff'].values[1:],
        linestyle='-',
        marker='o')
ax.set_xlabel('iteration')
ax.set_ylabel('Value')
plt.tight_layout()
# plt.savefig('outputs/rel_diff.png')
