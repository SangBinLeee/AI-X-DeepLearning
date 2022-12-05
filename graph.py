import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')

fig = plt.figure(figsize=(15, 9))
ax = fig.add_subplot(1, 1, 1)

#ax.plot(df['battery_power'], label='Battery_Power')
#ax.plot(df['clock_speed'], label='clock_speed')
#ax.plot(df['fc'], label='fc')
#ax.plot(df['mobile_wt'], label='mobile_wt')
#ax.plot(df['pc'], label='pc')
#ax.plot(df['talk_time'], label='talk_time')

#ax.plot(df['px_height'], label='px_height')
#ax.plot(df['px_width'], label='px_width')

#ax.plot(df['ram'], label='ram')

#ax.plot(df['sc_w'], label='sc_w')
#ax.plot(df['sc_h'], label='sc_h')

#ax.plot(df['m_dep'], label='m_dep')
#ax.plot(df['n_cores'], label='n_cores')

ax.plot(df['int_memory'], label='int_memory')

ax.set_title('Train Dataset')
ax.set_ylabel('Value')
ax.set_xlabel('Mobile')

ax.legend(fontsize=12, loc='best')
plt.show()