import matplotlib.pyplot as plt
import os 
from datetime import datetime


class LivePlot():
    def __init__(self):
        self.fig, self.ax = plt.subplot()
        self.ax.set_xlabel("epoch x 10")
        self.ax.set_ylabel("returns")
        self.ax.set_title("returns over epochs")
        self.data = None
        self.eps_data = None
        self.epochs = 0

    def update_plot(self, stats):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']
        self.epoch = len(self.data)
        self.ax.clear()
        self.ax.set_xlim(0,self.epochs)
        self.ax.plot(self.data, 'b-', label='returns')
        self.ax.plot(self.eps_data, 'r-', label='epsilon')
        self.ax.legend(loc='upper left')

        if not os.path.exists('plots'):
            os.makedirs('plots')
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.fig.savefig(f'plots/plot_{current_date}.png')
