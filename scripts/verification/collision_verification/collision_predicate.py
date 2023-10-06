import math
from math import atan2,atan,tan,cos
import verification.collision_verification.collision_verification_constants as constants
import numpy as np
import verification.collision_verification.model_reachability as model_reachability

# Input: 2 bounding boxes represented as a list of verticies corresponding to the covnex hull of bounding box 
# Output: List of verticies of convex hull of minkowski sum
def minkowski_sum_2d_convex_hull(bounding_box_verticies_1,bounding_box_verticies_2):
    # creates vertex polar angle tuple in form (vertex, polar_angle)
    polar_coordinate_pair_1 = [((
        vert,
        (atan2(vert[1],vert[0]) if vert[0] != 0 else 0)
        )) for vert in bounding_box_verticies_1]
    polar_coordinate_pair_2 = [((
        vert,
        (atan2(vert[1],vert[0]) if vert[0] != 0 else 0)
        )) for vert in bounding_box_verticies_2]
    
    # sort by polar angle
    polar_coordinate_pair_1.sort(key = lambda x:x[1])
    polar_coordinate_pair_2.sort(key = lambda x:x[1])

    # creates tuple list ((array_index, item_index), (vertex, polar_angle))
    vertex_with_index_1 = [ ([1,idx], polar_coordinate_pair_1[idx]) for idx in range(len(polar_coordinate_pair_1))]
    vertex_with_index_2 = [ ([2,idx], polar_coordinate_pair_2[idx]) for idx in range(len(polar_coordinate_pair_2))]
    polar_coordinate_pair_array = vertex_with_index_1 + vertex_with_index_2

    # Sort combined pair by angle
    polar_coordinate_pair_array.sort(key = lambda x:x[1][1])
    current_coordinate = (polar_coordinate_pair_1[0][0][0]+polar_coordinate_pair_2[0][0][0],polar_coordinate_pair_1[0][0][1]+polar_coordinate_pair_2[0][0][1])
    minkowski_sum_vertex_array = []

    # Follows planar case algorithm in https://en.wikipedia.org/wiki/Minkowski_addition
    for idx in range(len(polar_coordinate_pair_array)):
        array_identifier, polar_coordinate_pair = polar_coordinate_pair_array[idx]
        if array_identifier[0] == 1:
            vertex_index = array_identifier[1]+1
            if(vertex_index>=len(polar_coordinate_pair_1)):
                vertex_index = 0
            next_coordinate_pair = polar_coordinate_pair_1[vertex_index]
            dx,dy = (next_coordinate_pair[0][0] - polar_coordinate_pair[0][0], next_coordinate_pair[0][1] - polar_coordinate_pair[0][1])
            current_coordinate = (current_coordinate[0]+dx, current_coordinate[1]+dy)
            minkowski_sum_vertex_array.append(current_coordinate)
        if array_identifier[0] == 2:
            vertex_index = array_identifier[1]+1
            if(vertex_index>=len(polar_coordinate_pair_2)):
                vertex_index = 0
            next_coordinate_pair = polar_coordinate_pair_2[vertex_index]
            dx,dy = (next_coordinate_pair[0][0] - polar_coordinate_pair[0][0], next_coordinate_pair[0][1] - polar_coordinate_pair[0][1])
            current_coordinate = (current_coordinate[0]+dx, current_coordinate[1]+dy)
            minkowski_sum_vertex_array.append(current_coordinate)
    return minkowski_sum_vertex_array


def convex_hull_vertex_array_to_linear_constraint(convex_hull_array):
    # CX <= D
    # where X = [x,y]^T
    C = np.zeros((0,2))
    d = np.zeros((0,1))
    for idx in range(len(convex_hull_array)):
        C_i = np.zeros((1,2))
        d_i = np.zeros((1,1))
        x1 = convex_hull_array[idx][0]
        y1 = convex_hull_array[idx][1]
        x2 = 0
        y2 = 0
        if(idx == len(convex_hull_array)-1):
            x2 = convex_hull_array[0][0]
            y2 = convex_hull_array[0][1]
        else:
            x2 = convex_hull_array[idx+1][0]
            y2 = convex_hull_array[idx+1][1]
        C_i[0][0] = -(y2-y1)
        C_i[0][1] = x2-x1
        d_i[0][0] = -(y1 * (x2-x1) - x1 * (y2-y1))
        C = np.vstack([C,C_i])
        d = np.vstack([d,d_i])
    return C,d


def car_bounding_box_verticies(width, length, angle):
    x1 = (length/2) * math.cos(angle) - (width/2) * math.sin(angle)
    y1 = (width/2) * math.cos(angle) + (length/2) * math.sin(angle)
    x2 = -(length/2) * math.cos(angle) - (width/2) * math.sin(angle)
    y2 = (width/2) * math.cos(angle) - (length/2) * math.sin(angle)
    x3 = (length/2) * math.cos(angle) + (width/2) * math.sin(angle)
    y3 = -(width/2) * math.cos(angle) + (length/2) * math.sin(angle)
    x4 = -(length/2) * math.cos(angle) + (width/2) * math.sin(angle)
    y4 = -(width/2) * math.cos(angle) - (length/2) * math.sin(angle)
    car_bounding_box_verticies = [(x1,y1),(x2,y2),(x3,y3), (x4,y4)]
    return car_bounding_box_verticies


# Estimate Heading of both vehicles based on antiderivative of kinematic bicycle model
def predicted_headings(k,reachability_dt,pose_data,V_pi,zeta_pi,V_omega,zeta_omega):
	B_pi = atan(tan(zeta_pi) / 2)
	B_omega = atan(tan(zeta_omega) / 2)
	theta_omega = atan2(pose_data[3],pose_data[2])
	H_pi = math.pi/2 + (V_pi*tan(zeta_pi)*cos(B_pi))/constants.WHEELBASE * k  * reachability_dt
	H_omega = theta_omega + (V_omega*tan(zeta_omega)*cos(B_omega))/constants.WHEELBASE * k  * reachability_dt
	return H_pi, H_omega


def two_car_predicate(k,reachability_dt,pose_data,V_pi,zeta_pi,V_omega,zeta_omega):
    angle_pi, angle_omega = predicted_headings(k,reachability_dt,pose_data,V_pi,zeta_pi,V_omega,zeta_omega)
    car_pi = car_bounding_box_verticies(constants.PI_CAR_WIDTH, constants.PI_CAR_LENGTH, angle_pi)
    car_omega = car_bounding_box_verticies(constants.OMEGA_CAR_WIDTH, constants.OMEGA_CAR_LENGTH, angle_omega)
    minkowski_sum = minkowski_sum_2d_convex_hull(car_pi,car_omega)
    C_s,d_s = convex_hull_vertex_array_to_linear_constraint(minkowski_sum)
    C = np.hstack([C_s, np.zeros((C_s.shape[0],2)),-1*C_s,np.zeros((C_s.shape[0],2))])
    d=d_s   
    return C,d

# 
def car_pi_within_bounds_of_car_omega(k,reachability_dt,pose_data,V_pi,zeta_pi,V_omega,zeta_omega):
    angle_pi, angle_omega = predicted_headings(k,reachability_dt,pose_data,V_pi,zeta_pi,V_omega,zeta_omega)
    car_omega = car_bounding_box_verticies(constants.OMEGA_CAR_WIDTH, constants.OMEGA_CAR_LENGTH, angle_omega)
    C_s,d_s = convex_hull_vertex_array_to_linear_constraint(car_omega)
    C = np.hstack([C_s, np.zeros((C_s.shape[0],2)),-1*C_s,np.zeros((C_s.shape[0],2))])
    d=d_s   
    return C,d

def predicate_next_k_timesteps(k,reachability_dt,X_0,U_0):
    V_pi,zeta_pi,V_omega,zeta_omega = U_0
    predicates = []
    for reachability_timestep_idx in range(k):
        H,g = two_car_predicate(reachability_timestep_idx,reachability_dt,X_0[4:8],V_pi,zeta_pi,V_omega,zeta_omega)
        predicates.append([H,g])
    return predicates

# def predicate_next_k_timesteps(k,reachability_dt,X_0,sigma_0,U_0):
#     predicates = []

#     model_dt = reachability_dt / 10
#     V_pi,zeta_pi,V_omega,zeta_omega = U_0
#     # Convert Initial State to Probstar 
#     c = (np.expand_dims(X_0, axis=0)).transpose()
#     V = np.diag(sigma_0)

#     # Assemble probstars k = reachability_start_idx to k = k_max
#     for reachability_timestep_idx in range(k):
#         for model_timestep_idx in range(10):
#             A = model_reachability.two_car_A(V_pi,zeta_pi,V_omega,zeta_omega,model_dt)
#             c = np.matmul(A,c)
#             V = np.matmul(A, V)

#         # Apply Collision Bounding Predicate
#         H,g = two_car_predicate(reachability_timestep_idx,reachability_dt,X_0[4:8],V_pi,zeta_pi,V_omega,zeta_omega)
#         C = np.matmul(H,V)
#         d = g-np.matmul(H,c)
#         predicates.append([np.asarray(C),np.asarray(d)])
#     return predicates