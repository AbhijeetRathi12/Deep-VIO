import math
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

## circle generation 
def circle(R, center):
    pose = []
    thetas = np.linspace(0, 2*np.pi, 10)
    last_pose= [0,0,0,0,0,0]
    for theta in thetas:
        x = R * math.cos(theta) + center[0]
        y = R * math.sin(theta) + center[1]
        z = center[2] 
        yaw = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(0, math.pi/4)
        roll = random.uniform(0, math.pi/4)  
        currrent_pose = [yaw, pitch, roll, x, y , z]
        del_x = x - last_pose[3]
        del_y = y - last_pose[4]
        del_z = z - last_pose[5]
        last_pose = currrent_pose          
        pose.append([2 ,yaw, pitch, roll, del_x, del_y, del_z, 1, 1] )

    pose = np.array(pose)

    return pose


def figure_eight(a, z):
    t = np.linspace(0, 2*np.pi, 30)
    last_pose = [0,0,0,0,0,0]
    pose = []
    for i in t:
        x = a * math.sin(i)
        y = a * math.sin(i) * math.cos(i)
        yaw = random.uniform(0, 2 * math.pi)
        pitch = random.uniform(0, math.pi/4)
        roll = random.uniform(0, math.pi/4)  
        currrent_pose = [yaw, pitch, roll, x, y , z]
        del_x = x - last_pose[3]
        del_y = y - last_pose[4]
        del_z = z - last_pose[5]
        del_yaw = yaw - last_pose[0]
        del_pitch = pitch - last_pose[1]
        del_roll = roll - last_pose[2] 
        last_pose = currrent_pose          
        pose.append( [2 ,yaw, pitch, roll, del_x, del_y, del_z, 1, 1] ) 

    pose = np.array(pose)    
    return pose  

def main():

    pose = figure_eight(5,10)
    csv_file = 'data.csv'
    # Write the data to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pose)
    
    pose_array = pose
    x = pose_array[:,4]
    y = pose_array[:,5]
    z = pose_array[:,6] 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z,c=z,  cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()           
             

   
if __name__ == "__main__":
    main()