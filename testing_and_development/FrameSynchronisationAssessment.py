import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
class CheckSyncing():

    def __init__(self):
        self.filename = []
        self.file_root = []
        self.data = {'side': [], 'front': [], 'overhead': []}
        self.data_diff = {'side': [], 'front': [], 'overhead': []}

    def run(self):
        print("Running data synchronization check...")
        self.load_data()
        print("Data loaded.")
        self.clean_data()
        print("Data cleaned.")
        self.plot_frame_diff_within_cam()
        print("First plot created.")
        self.plot_frame_diff_between_cam()
        print("Second plot created.")
        plt.show()

    def load_data(self):
        # to show only the dialog without any other GUI elements
        root = tk.Tk()
        root.withdraw()

        # get file path for side camera
        file_path_side = filedialog.askopenfilename()

        # from side path, get front and overhead paths
        file_path_front = file_path_side.replace('side', 'front')
        file_path_overhead = file_path_side.replace('side', 'overhead')

        # get common file name
        self.file_name = '_'.join(file_path_side.split('/')[-1].split('_')[0:5])
        self.file_root = '/'.join(file_path_side.split('/')[:-1])

        # load data
        self.data = {'side': pd.read_csv(file_path_side),
                'front': pd.read_csv(file_path_front),
                'overhead': pd.read_csv(file_path_overhead)}

    def clean_data(self):
        for cam in ['side', 'front', 'overhead']:
            self.data[cam] = self.data[cam]['Timestamp'] - self.data[cam]['Timestamp'][0]

    def plot_frame_diff_within_cam(self):
        fig, ax = plt.subplots(4, 1, figsize=(45, 25))
        colors = ['r', 'g', 'b']

        for i, cam in enumerate(['side', 'front', 'overhead']):
            self.data_diff[cam] = self.data[cam].diff()
            ax[i].plot(self.data_diff[cam].index, self.data_diff[cam].values, label=cam, color=colors[i])
            ax[i].set_title(f'{cam} camera')
            ax[i].set_ylim(0.25e7, 2.24e7)
        ax[3].plot(self.data_diff['side'].index, self.data_diff['side'].values, label='side', color='r', ls='-')
        ax[3].plot(self.data_diff['front'].index, self.data_diff['front'].values, label='front', color='g', ls='-.')
        ax[3].plot(self.data_diff['overhead'].index, self.data_diff['overhead'].values, label='overhead', color='b', ls='--')
        ax[3].set_title('All cameras')
        ax[3].set_ylim(0.25e7, 2.24e7)
        ax[3].legend()

        plt.suptitle('Frame difference within camera\n%s' %self.file_name)

        plt.savefig(f'H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Misc/FrameSyncingCheck/{self.file_name}_frame_diff_within_cam.png')


    def plot_frame_diff_between_cam(self):
        fig, ax = plt.subplots(3, 1, figsize=(45, 25))

        side_front = self.data['side'] - self.data['front']
        side_overhead = self.data['side'] - self.data['overhead']
        front_overhead = self.data['front'] - self.data['overhead']

        for i, diff in enumerate([side_front, side_overhead, front_overhead]):
            ax[i].plot(diff.index, diff.values, color='purple')
            ax[i].set_title(f'Difference between cameras: {["side_front", "side_overhead", "front_overhead"][i]}')

        plt.suptitle('Frame difference between cameras\n%s' %self.file_name)

        plt.savefig(f'H:/Dual-belt_APAs/analysis/DLC_DualBelt/Manual_Labelling/Misc/FrameSyncingCheck/{self.file_name}_frame_diff_between_cam.png')



if __name__ == '__main__':
    CheckSyncing().run()




