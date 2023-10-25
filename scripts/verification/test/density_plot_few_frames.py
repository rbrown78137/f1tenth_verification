import matplotlib
import verification.collision_verification.collision_probability as collision_probability
import matplotlib.pyplot as plt
import numpy as np
import verification.collision_verification.initial_state as initial_state
import verification.collision_verification.collision_verification_constants as constants
import matplotlib.lines as mlines
import cv2 as cv
import pickle
from multiprocessing import Pool
import verification.collision_verification.collision_verification_constants as constants
import time
import copy

CALCULATE_DENSITY = True

VIEW_X_MIN = 0 
VIEW_X_MAX = 0 
VIEW_Y_MIN = 0
VIEW_Y_MAX = 0

# if constants.RECORDING_NUMBER == 1:
#     VIEW_X_MIN = -.8
#     VIEW_X_MAX = 0.2 
#     VIEW_Y_MIN = -0.2 
#     VIEW_Y_MAX = 3.2

# VIEW_X_MIN += 1e-7
# VIEW_X_MAX += 1e-7
# VIEW_Y_MIN += 1e-7
# VIEW_Y_MAX += 1e-7

FUTURE_TIME_STEPS = 5
PROBABILITY_SQUARE_DISTANCE = 0.1
PROBABILITY_SQUARE_DISTANCE_REFINEMENT = 0.01 #0.01

PROB_THRESHOLD = 1e-4 * PROBABILITY_SQUARE_DISTANCE **2
CIRCLE_RADIUS = 0.02
RECTANGLE_WIDTH = 0.9 * CIRCLE_RADIUS
MARKER_SIZE = 10
COLORS = ["blue","red","green","yellow","purple"]
COLOR_CONSTANT = 1
COLOR_MAP = {"blue":[0,0,255],"red":[255,0,0],"green":[0,127,0],"yellow":[255,165,0],"purple":[127,0,127]}
# COLOR_MAP = {"blue":[255,255,255],"red":[255,255,255],"green":[255,255,255],"yellow":[255,255,255],"purple":[255,255,255]}
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)


multiprocessing_cores = 16
pool = Pool(multiprocessing_cores)
time.sleep(10)

def draw_centers(X_0, sigma_0, U_0, ax,draw_other_car=True):
    modified_U_0 = copy.deepcopy(U_0)
    # probstars = collision_probability.probstar_next_k_time_steps_given_initial_state(FUTURE_TIME_STEPS,0,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,modified_U_0,6)
    probstars = []
    for x in range(5):
        probstars.append(collision_probability.probstar_next_k_time_steps_given_initial_state(1,x,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,modified_U_0,6)[0])
    # Initial Position
    ax.plot([X_0[0]], [X_0[1]], marker="s", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)) 
    if draw_other_car:
        ax.plot([X_0[4]], [X_0[5]], marker="o", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0))
    # Future Time Steps
    for star_idx,star in enumerate(probstars):
        x_pi = star.V[0,0]
        y_pi = star.V[1,0]
        x_omega = star.V[4,0]
        y_omega = star.V[5,0]
        ax.plot([x_pi], [y_pi], marker="s", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)) # [x/255 for x in COLOR_MAP[COLORS[star_idx]]]
        if draw_other_car:
            ax.plot([x_omega], [y_omega], marker="o", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0))
        # ax.plot([x_pi], [y_pi], marker="s", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(COLOR_MAP[COLORS[star_idx]][0]/255,COLOR_MAP[COLORS[star_idx]][1]/255,COLOR_MAP[COLORS[star_idx]][2]/255)) # [x/255 for x in COLOR_MAP[COLORS[star_idx]]]
        # ax.plot([x_omega], [y_omega], marker="o", markersize=MARKER_SIZE, markeredgecolor=(0,0,0), markerfacecolor=(COLOR_MAP[COLORS[star_idx]][0]/255,COLOR_MAP[COLORS[star_idx]][1]/255,COLOR_MAP[COLORS[star_idx]][2]/255))



def get_probability_image(X_0, sigma_0, U_0,draw_other_car=True):
    image_width_large_pass = int((VIEW_X_MAX - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE)
    image_height_large_pass = int((VIEW_Y_MAX - VIEW_Y_MIN) / PROBABILITY_SQUARE_DISTANCE)

    image_width = int((VIEW_X_MAX - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT)
    image_height = int((VIEW_Y_MAX - VIEW_Y_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT)
    omega_image = np.ones((FUTURE_TIME_STEPS,image_height,image_width,3))
    pi_image = np.ones((FUTURE_TIME_STEPS,image_height,image_width,3))
    if CALCULATE_DENSITY:
        # Calculate Distribution for Car Omega
        if draw_other_car:
            for timestep_idx in range(0,FUTURE_TIME_STEPS):
                prob_map = np.zeros((image_height,image_width,3))
                REFINE_X_MIN = VIEW_X_MAX
                REFINE_X_MAX = VIEW_X_MIN
                REFINE_Y_MIN = VIEW_Y_MAX
                REFINE_Y_MAX = VIEW_Y_MIN
                refinement_probstars = []
                for i in range(image_width_large_pass):
                    for j in range(image_height_large_pass):
                        X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE
                        X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE
                        Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE
                        Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE
                        probstars = collision_probability.car_omega_probstar_next_k_time_steps(X_MIN,X_MAX,Y_MIN,Y_MAX,FUTURE_TIME_STEPS,0,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,U_0,6)
                        probstar = probstars[timestep_idx]
                        refinement_probstars.append(probstar)

                refinement_probs = pool.map(collision_probability.estimate_probstar_probability,refinement_probstars)
                
                for i in range(image_width_large_pass):
                    for j in range(image_height_large_pass):
                        X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE
                        X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE
                        Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE
                        Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE
                        prob = refinement_probs.pop(0)
                        if prob > PROB_THRESHOLD:
                            if REFINE_X_MAX < X_MAX:
                                REFINE_X_MAX = X_MAX
                            if REFINE_Y_MAX < Y_MAX:
                                REFINE_Y_MAX = Y_MAX
                            if REFINE_X_MIN > X_MIN:
                                REFINE_X_MIN = X_MIN
                            if REFINE_Y_MIN > Y_MIN:
                                REFINE_Y_MIN = Y_MIN
                i_start =int( (REFINE_X_MIN - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
                i_end =int( (REFINE_X_MAX - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
                j_start =int( (VIEW_Y_MAX -REFINE_Y_MAX) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
                j_end =int( (VIEW_Y_MAX -REFINE_Y_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
                
                small_probstars = []
                for i in range(i_start,i_end,1):
                    for j in range(j_start,j_end,1):
                        X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                        X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                        Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                        Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                        probstars = collision_probability.car_omega_probstar_next_k_time_steps(X_MIN,X_MAX,Y_MIN,Y_MAX,FUTURE_TIME_STEPS,0,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,U_0,6)
                        probstar = probstars[timestep_idx]
                        small_probstars.append(probstar) 
                small_probs = pool.map(collision_probability.estimate_probstar_probability,small_probstars)
                        
                for i in range(i_start,i_end,1):
                    for j in range(j_start,j_end,1):
                        prob = small_probs.pop(0)
                        if i < image_width and j<image_height:
                            prob_map[j,i,0] = prob
                            prob_map[j,i,1] = prob
                            prob_map[j,i,2] = prob

                prob_max = prob_map.max()
                if prob_max > 0:
                    prob_map /= prob_max
                color_array = np.zeros((image_height,image_width,3))
                color_array[...,0] = COLOR_MAP[COLORS[timestep_idx]][0]
                color_array[...,1] = COLOR_MAP[COLORS[timestep_idx]][1]
                color_array[...,2] = COLOR_MAP[COLORS[timestep_idx]][2]
                omega_image[timestep_idx] = (color_array + (1-prob_map*COLOR_CONSTANT) * (255-color_array)) / 255

        # Calculate Distribution for Car Pi
        for timestep_idx in range(0,FUTURE_TIME_STEPS):
            prob_map = np.zeros((image_height,image_width,3))
            REFINE_X_MIN = VIEW_X_MAX
            REFINE_X_MAX = VIEW_X_MIN
            REFINE_Y_MIN = VIEW_Y_MAX
            REFINE_Y_MAX = VIEW_Y_MIN
            pi_refinement_probstars = []
            for i in range(image_width_large_pass):
                for j in range(image_height_large_pass):
                    X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE
                    X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE
                    Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE
                    Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE
                    probstars = collision_probability.car_pi_probstar_next_k_time_steps(X_MIN,X_MAX,Y_MIN,Y_MAX,FUTURE_TIME_STEPS,0,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,U_0,6)
                    probstar = probstars[timestep_idx]
                    pi_refinement_probstars.append(probstar)

            pi_refinement_probs = pool.map(collision_probability.estimate_probstar_probability,pi_refinement_probstars)
            for i in range(image_width_large_pass):
                for j in range(image_height_large_pass):
                    X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE
                    X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE
                    Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE
                    Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE
                    prob = pi_refinement_probs.pop(0)
                    if prob > PROB_THRESHOLD:
                        if REFINE_X_MAX < X_MAX:
                            REFINE_X_MAX = X_MAX
                        if REFINE_Y_MAX < Y_MAX:
                            REFINE_Y_MAX = Y_MAX
                        if REFINE_X_MIN > X_MIN:
                            REFINE_X_MIN = X_MIN
                        if REFINE_Y_MIN > Y_MIN:
                            REFINE_Y_MIN = Y_MIN

            i_start =int( (REFINE_X_MIN - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
            i_end =int( (REFINE_X_MAX - VIEW_X_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
            j_start =int( (VIEW_Y_MAX -REFINE_Y_MAX) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
            j_end =int( (VIEW_Y_MAX -REFINE_Y_MIN) / PROBABILITY_SQUARE_DISTANCE_REFINEMENT )
            
            pi_small_probstars = []
            for i in range(i_start,i_end,1):
                for j in range(j_start,j_end,1):
                    X_MIN = VIEW_X_MIN + i * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                    X_MAX = VIEW_X_MIN + (i+1) * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                    Y_MIN = VIEW_Y_MAX - (j+1) * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                    Y_MAX = VIEW_Y_MAX - j * PROBABILITY_SQUARE_DISTANCE_REFINEMENT
                    probstars = collision_probability.car_pi_probstar_next_k_time_steps(X_MIN,X_MAX,Y_MIN,Y_MAX,FUTURE_TIME_STEPS,0,constants.REACHABILITY_DT,constants.MODEL_SUBTIME_STEPS,X_0,sigma_0,U_0,6)
                    probstar = probstars[timestep_idx]
                    pi_small_probstars.append(probstar) 

            pi_small_probs = pool.map(collision_probability.estimate_probstar_probability,pi_small_probstars)
                    
            for i in range(i_start,i_end,1):
                for j in range(j_start,j_end,1):
                    prob = pi_small_probs.pop(0)
                    if i < image_width and j<image_height:
                        prob_map[j,i,0] = prob
                        prob_map[j,i,1] = prob
                        prob_map[j,i,2] = prob
            prob_max = prob_map.max()
            if prob_max > 0:
                prob_map /= prob_max
            color_array = np.zeros((image_height,image_width,3))
            color_array[...,0] = COLOR_MAP[COLORS[timestep_idx]][0]
            color_array[...,1] = COLOR_MAP[COLORS[timestep_idx]][1]
            color_array[...,2] = COLOR_MAP[COLORS[timestep_idx]][2]
            pi_image[timestep_idx] = (color_array + (1-prob_map*COLOR_CONSTANT) * (255-color_array)) / 255
    final_image = np.minimum(omega_image.prod(axis=0),pi_image.prod(axis=0))
    return final_image

def get_graph_instance(pose_history,actuation_history,pose_time_history,reachability_dt=0.25,model_sub_time_steps=10):
    draw_other_car = True
    if len(pose_history[0])==0:
        draw_other_car = False
    X_0, sigma_0, U_0 = initial_state.initial_state(pose_history,actuation_history,pose_time_history)
    final_image = get_probability_image(X_0,sigma_0,U_0,draw_other_car)

    fig = plt.figure()

    ax = fig.add_subplot()
    legend_elements = [#mlines.Line2D([0], [0], color=COLORS[0], lw=2, label='1 Future Step'),
                       #mlines.Line2D([0], [0], color=COLORS[1], lw=2, label='2 Future Steps'),
                       #mlines.Line2D([0], [0], color=COLORS[2], lw=2, label='3 Future Steps'),
                       #mlines.Line2D([0], [0], color=COLORS[3], lw=2, label='4 Future Steps'),
                       #mlines.Line2D([0], [0], color=COLORS[4], lw=2, label='5 Future Steps'),
                       mlines.Line2D([0], [0], color=(0,0,0), marker='s', markersize=12, lw=0, label='Ego Vehicle',markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0)),
                       mlines.Line2D([0], [0], color=(0,0,0), marker='o', markersize=12, lw=0, label='Other Vehicle',markeredgecolor=(0,0,0), markerfacecolor=(1.0,1.0,1.0))
                       ]
    ax.legend(handles=legend_elements,loc="upper right")
    fig.set_figheight(6)
    fig.set_figwidth(6)
    # Display the image
    ax.imshow(final_image,aspect='auto',extent=(VIEW_X_MIN,VIEW_X_MAX,VIEW_Y_MIN,VIEW_Y_MAX))
    draw_centers(X_0, sigma_0, U_0, ax, draw_other_car)
    ax.set_title("Distributions for Predicted Paths",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17})
    ax.set_ylabel("Y (meters)",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17})
    ax.set_xlabel("X (meters)",fontdict={'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17})
    fig.subplots_adjust(
        top=0.95,
        bottom=0.11,
        left=0.15,
        right=0.95,
        hspace=0.2,
        wspace=0.2
    )
    # fig.canvas.draw()
    plt.show()
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img = cv.resize(img,(480,480))
    # img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
    # cv.imshow(winname="Test",mat=img)
    # cv.waitKey(30)
    return None
    # cv.imshow(winname="Test",mat=img)
    # cv.waitKey(0)

def get_VIEW(idx):
    global VIEW_X_MIN,VIEW_X_MAX,VIEW_Y_MIN,VIEW_Y_MAX

    VIEW_X_MIN = -0.7
    VIEW_X_MAX = 0.7
    VIEW_Y_MIN = -0.2
    VIEW_Y_MAX = 3

    if idx == 1:
        VIEW_X_MIN = -.3
        VIEW_X_MAX = 0.2 
        VIEW_Y_MIN = -0.2 
        VIEW_Y_MAX = 2.5

    if idx == 2:
        VIEW_X_MIN = -.2
        VIEW_X_MAX = 0.8 
        VIEW_Y_MIN = -0.2 
        VIEW_Y_MAX = 2.5
    
    if idx == 3:
        VIEW_X_MIN = -.8
        VIEW_X_MAX = 0.2 
        VIEW_Y_MIN = -0.2 
        VIEW_Y_MAX = 3.2

    if idx == 4:
        VIEW_X_MIN = -.4
        VIEW_X_MAX = 0.6 
        VIEW_Y_MIN = -0.2 
        VIEW_Y_MAX = 3
    
    if idx == 5:
        VIEW_X_MIN = -0.7
        VIEW_X_MAX = 0.7
        VIEW_Y_MIN = -0.2
        VIEW_Y_MAX = 3
    
    if idx == 6:
        VIEW_X_MIN = -0.7
        VIEW_X_MAX = 0.7
        VIEW_Y_MIN = -0.2
        VIEW_Y_MAX = 3

    if idx == 7:
        VIEW_X_MIN = -0.2
        VIEW_X_MAX = 1
        VIEW_Y_MIN = -0.2
        VIEW_Y_MAX = 3

    if idx == 8:
        VIEW_X_MIN = -0.2
        VIEW_X_MAX = 1
        VIEW_Y_MIN = -0.2
        VIEW_Y_MAX = 3


    # VIEW_X_MIN += 1e-7
    # VIEW_X_MAX += 1e-7
    # VIEW_Y_MIN += 1e-7
    # VIEW_Y_MAX += 1e-7


if __name__ == "__main__":
    for i in range(7,8):
        get_VIEW(i)
        with open('saved_data/new_video/frame_history_'+str(i)+'.pkl', 'rb') as f:
            history = pickle.load(f)
            for frame_idx,frame_data in enumerate(history):
                key_frames = [160,310,350]
                # Key frames From Videos 
                # Video 1: 100, 170, 350
                # Video 5: 
                # Video 7: 160, 310, 350
                if not(frame_idx in key_frames):
                    continue
                # if frame_idx > 1:
                #     continue
                pose_history = frame_data[1][0]
                actuation_history = frame_data[1][1]
                pose_time_history = frame_data[1][2]
                frame = get_graph_instance(pose_history,actuation_history,pose_time_history)
                print(f"Got frame {frame_idx} out of {len(history)}")
