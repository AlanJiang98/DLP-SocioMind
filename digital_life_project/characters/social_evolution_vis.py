"""
Author: Jianping Jiang (alanjjp98@gmail.com)

File: social_evolution_vis.py
Description: Visualize the social evolution of two characters
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

base_font_size = 12


plt.rcParams["font.family"] = "Times New Roman"

episodes = ["1 First meet", "2 Poor exam results", "3 Game & time", "4 Photo memory", "5 Taking photos"]

sns.set_palette("pastel")


P = [2, 2, 5, 4, 7, ]
A = [7, 5, 6, 5, 6, ]
D = [3, 3, 4, 3, 4, ]

I = [2, 1, 3, 4, 5, ]
T = [1, 1, 2, 3, 4, ]
S = [2, 1, 2, 3, 5, ]


data = {
    'Episode': episodes,
    'Intimacy': I,
    'Trust': T,
    'Supportiveness': S,
    'Pleasure': P,
    'Arousal': A,
    'Dominance': D,
}


custom_palette = ['#F8BBD0', '#90CAF9', '#A5D6A7', '#FFF59D', '#FDBA96', '#9C27B0']


df = pd.DataFrame(data)
melted_df = df.melt(id_vars='Episode', var_name='Dimension', value_name='Rating')


plt.rcParams['font.family'] = 'Times New Roman'
sns.set_palette("pastel")
social_relationships = melted_df[melted_df['Dimension'].isin(['Intimacy', 'Trust', 'Supportiveness'])]
emotions = melted_df[melted_df['Dimension'].isin(['Pleasure', 'Arousal', 'Dominance'])]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

width = 0.7

sns.barplot(x='Episode', y='Rating', hue='Dimension', data=emotions, ax=ax1, palette=custom_palette[3:], width=width)
# ax1.set_title('Social Relationships over Episodes')
ax1.set_xlabel('')
ax1.set_ylabel('Rating', fontsize=base_font_size)
ax1.legend(title='Emotion', loc='upper left', fontsize=base_font_size-2)

sns.barplot(x='Episode', y='Rating', hue='Dimension', data=social_relationships, ax=ax2, palette=custom_palette[:3], width=width)
# ax2.set_title('Emotions over Episodes')
ax2.set_xlabel('', fontsize=base_font_size)

ax2.set_xticklabels(episodes, fontsize=base_font_size)

ax2.set_ylabel('Rating', fontsize=base_font_size)
ax2.legend(title='Social relationship', loc='upper left', fontsize=base_font_size-2)

plt.tight_layout()
# plt.show()
plt.savefig('social_evolution.svg', format='svg', dpi=600)