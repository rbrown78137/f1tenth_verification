import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import cv2 as cv
import pickle
from multiprocessing import Pool
import verification.collision_verification.collision_probability as collision_probability
import verification.collision_verification.initial_state as initial_state
import verification.collision_verification.collision_verification_constants as constants

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)

RECORDING = 3
CLIP_TIME_DIFF = 0
FUTURE_TIME_STEPS = 5

if RECORDING == 3:
    CLIP_TIME_DIFF = 0

if __name__ == "__main__":
    ground_truth_data = None
    prediction_data = None

    with open('saved_data/ground_truth_pose.pkl','rb') as f:
        ground_truth_data = pickle.load(f)

    with open('saved_data/old_video/frame_history_3.pkl','rb') as f:
        prediction_data = pickle.load(f)

    ground_truth_x_positions = []
    ground_truth_y_positions = []
    
    idx_of_interest = 0
    time_of_prediction = prediction_data[idx_of_interest][0]

    X_0, sigma_0, U_0 = initial_state.initial_state(prediction_data[idx_of_interest][1][0],prediction_data[idx_of_interest][1][1],prediction_data[idx_of_interest][1][2])


    for x in range(5):
        probstar = collision_probability.probstar_next_k_time_steps_given_initial_state(1,x,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,U_0,6)[0])

    # x_prediction = prediction_data[0][1][0][0][0]
    # y_prediction = prediction_data[0][1][0][0][1]

    gt_time_idx_pairs = [(ground_truth_data[idx][0]-time_of_prediction,idx) for idx in range(len(ground_truth_data))]
    gt_time_idx_pairs.sort(key=lambda x: x[0])
    corresponding_gt_data = ground_truth_data[gt_time_idx_pairs[0][1]]
    x_ground_truth = ground_truth_data[1]
    y_ground_truth = ground_truth_data[1]



    fig = plt.figure()
    ax = fig.add_subplot()

    ax.plot([1], [1], marker="s", markersize=10, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)) 
    ax.plot([1], [1], marker="o", markersize=10, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)) 

    legend_elements = [
                       mlines.Line2D([0], [0], color=(0,0,0), marker='s', markersize=12, lw=0, label='Ego Vehicle',markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)),
                       mlines.Line2D([0], [0], color=(0,0,0), marker='o', markersize=12, lw=0, label='Other Vehicle',markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0))
                       ]

    ax.legend(handles=legend_elements,loc="upper right")

    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax.set_title("Ground Truth Vs Model Prediction",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18})
    ax.set_ylabel("Y",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18})
    ax.set_xlabel("X",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18})
    
    # fig.canvas.draw()
    plt.plot()
    plt.close('all')
