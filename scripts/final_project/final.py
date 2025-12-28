###
# Final project with MuJoCo
# SI100B Robotics Programming
# This code is modified based on the MuJoCo template code at https://github.com/pab47/pab47.github.io/tree/master.
# Date: Dec., 2025
###

import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os

xml_path = '../../models/universal_robots_ur5e/scene.xml' #xml file (assumes this is in the same folder as this file)
#################################
## USER CODE: Set simulation parameters here
#################################
simend = 114514 #simulation time (second)
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)
#################################

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# Helper function
def IK_controller(model, data, X_ref, q_pos):
    # Compute Jacobian
    position_Q = data.site_xpos[0]

    jacp = np.zeros((3, 6))
    mj.mj_jac(model, data, jacp, None, position_Q, 7)

    J = jacp.copy()
    Jinv = np.linalg.pinv(J)

    # Reference point
    X = position_Q.copy()
    dX = X_ref - X

    # Compute control input
    dq = Jinv @ dX

    return q_pos + dq

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

# Get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1920, 1080, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)


########################################
## USER CODE: Set camera view here
########################################
# Example on how to set camera configuration
cam.azimuth =  85.66333333333347 
cam.elevation =  -35.33333333333329
cam.distance =  2.22
cam.lookat = np.array([ -0.09343103051557476 , 0.31359595076587915 , 0.22170312166086661 ])
########################################

# Initialize the controller
init_controller(model,data)

# Set the controller
mj.set_mjcb_control(controller)



traj_points = []
MAX_TRAJ = 5e5  # Maximum number of trajectory points to store
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])

######################################
## USER CODE STARTS HERE
######################################
strokes = [[np.array([-0.3947, 0.3497]), np.array([-0.381, 0.3337]), np.array([-0.3683, 0.2797])],
[np.array([-0.3837, 0.3517]), np.array([-0.382, 0.3493]), np.array([-0.369, 0.346]), np.array([-0.302, 0.3607]), np.array([-0.2863, 0.36]), np.array([-0.275, 0.3493]), np.array([-0.286, 0.318]), np.array([-0.2947, 0.3147])],
[np.array([-0.3613, 0.286]), np.array([-0.358, 0.2907]), np.array([-0.3003, 0.302]), np.array([-0.283, 0.3027]), np.array([-0.275, 0.3007])],
[np.array([-0.3963, 0.2463]), np.array([-0.3833, 0.2437]), np.array([-0.3697, 0.2443]), np.array([-0.276, 0.2623]), np.array([-0.2603, 0.262])],
[np.array([-0.447, 0.187]), np.array([-0.4287, 0.1837]), np.array([-0.379, 0.192]), np.array([-0.2393, 0.2067]), np.array([-0.2223, 0.204]), np.array([-0.21, 0.1997])],
[np.array([-0.3477, 0.238]), np.array([-0.3347, 0.2293]), np.array([-0.3347, 0.2233]), np.array([-0.3403, 0.189]), np.array([-0.3543, 0.1553]), np.array([-0.3637, 0.1423]), np.array([-0.377, 0.129]), np.array([-0.4013, 0.1147]), np.array([-0.426, 0.105]), np.array([-0.4347, 0.1037])],
[np.array([-0.3283, 0.1887]), np.array([-0.3243, 0.1807]), np.array([-0.321, 0.179]), np.array([-0.303, 0.1573]), np.array([-0.2807, 0.135]), np.array([-0.2577, 0.116]), np.array([-0.238, 0.109]), np.array([-0.1923, 0.1003])],
[np.array([-0.1253, 0.3497]), np.array([-0.1143, 0.3377]), np.array([-0.1113, 0.328]), np.array([-0.1097, 0.2737]), np.array([-0.117, 0.176]), np.array([-0.124, 0.1427]), np.array([-0.1243, 0.121])],
[np.array([-0.1027, 0.343]), np.array([-0.095, 0.3377]), np.array([0.036, 0.3613]), np.array([0.048, 0.357]), np.array([0.0573, 0.3473]), np.array([0.0567, 0.3273]), np.array([0.0637, 0.17]), np.array([0.0627, 0.1307]), np.array([0.057, 0.1167]), np.array([0.0543, 0.1167]), np.array([0.0377, 0.1197]), np.array([0.0083, 0.1307])],
[np.array([-0.0743, 0.2923]), np.array([-0.071, 0.2907]), np.array([-0.0537, 0.2903]), np.array([-0.0083, 0.301]), np.array([0.0093, 0.3013])],
[np.array([-0.08, 0.2477]), np.array([-0.0717, 0.2403]), np.array([-0.0697, 0.2353]), np.array([-0.0597, 0.181])],
[np.array([-0.0613, 0.24]), np.array([-0.0557, 0.2443]), np.array([-0.0197, 0.253]), np.array([-0.0127, 0.2543]), np.array([-0.0047, 0.2517]), np.array([0.0027, 0.2443]), np.array([-0.0057, 0.2163]), np.array([-0.0127, 0.211])],
[np.array([-0.0533, 0.1933]), np.array([-0.0487, 0.197]), np.array([-0.0157, 0.2027]), np.array([-0.003, 0.2027]), np.array([0.005, 0.2])],
[np.array([0.288, 0.3147]), np.array([0.3053, 0.311]), np.array([0.4327, 0.329]), np.array([0.4463, 0.3287]), np.array([0.4543, 0.3263])],
[np.array([0.3193, 0.3573]), np.array([0.3333, 0.3437]), np.array([0.334, 0.3407]), np.array([0.3407, 0.2743]), np.array([0.3363, 0.2687])],
[np.array([0.4003, 0.373]), np.array([0.407, 0.3667]), np.array([0.4123, 0.3573]), np.array([0.397, 0.28]), np.array([0.3903, 0.2737])],
[np.array([0.2357, 0.2513]), np.array([0.2547, 0.2477]), np.array([0.3467, 0.2607]), np.array([0.474, 0.2717]), np.array([0.492, 0.2693]), np.array([0.5053, 0.264])],
[np.array([0.2913, 0.226]), np.array([0.302, 0.217]), np.array([0.3197, 0.145])],
[np.array([0.3113, 0.224]), np.array([0.3167, 0.2207]), np.array([0.4097, 0.2343]), np.array([0.4287, 0.2323]), np.array([0.439, 0.2197]), np.array([0.431, 0.1997]), np.array([0.417, 0.152])],
[np.array([0.332, 0.1903]), np.array([0.3837, 0.1977]), np.array([0.401, 0.196])],
[np.array([0.356, 0.253]), np.array([0.3647, 0.248]), np.array([0.368, 0.2393]), np.array([0.367, 0.1693]), np.array([0.3617, 0.1623])],
[np.array([0.326, 0.148]), np.array([0.3323, 0.1513]), np.array([0.4027, 0.1597]), np.array([0.409, 0.162])],
[np.array([0.3463, 0.1247]), np.array([0.3337, 0.1247]), np.array([0.3197, 0.112]), np.array([0.299, 0.0983]), np.array([0.2667, 0.085])],
[np.array([0.3993, 0.1337]), np.array([0.4383, 0.1087]), np.array([0.449, 0.0967]), np.array([0.4537, 0.086])],
]

# Initialize joint configuration
init_qpos = np.array([0.6353559, -1.28588984, 2.14838487, -2.61087434, -1.5903009, -0.06818645])

data.qpos[:] = init_qpos
cur_q_pos = init_qpos.copy()

# 构建完整轨迹：包含抬笔、落笔、书写、再抬笔
trajectory = []
z_write = 0.1   # 书写高度
z_lift = 0.15    # 抬笔高度

for i, stroke in enumerate(strokes):
    # 每个 stroke 为可变长度的点列表：[(x0,y0), (x1,y1), ..., (xn,yn)]
    if not stroke:
        continue

    sx, sy = stroke[0]
    ex, ey = stroke[-1]

    if i == 0:
        # 第一笔：从初始位置抬着走到起始点上方
        trajectory.append(np.array([sx, sy, z_lift]))   # 移动到起始点上方（抬笔状态）
    else:
        # 从上一笔结束点上方（z=0.2）移动到当前笔画起始点上方
        prev_end = strokes[i-1][-1]
        trajectory.append(np.array([prev_end[0], prev_end[1], z_lift]))
        trajectory.append(np.array([sx, sy, z_lift]))

    # 落笔：在第一个点处落笔
    trajectory.append(np.array([sx, sy, z_write]))

    # 书写：依次经过子列表中间的所有点，直到最后一个点
    for px, py in stroke[1:]:
        trajectory.append(np.array([px, py, z_write]))

    # 抬笔：在最后一个点上抬起
    trajectory.append(np.array([ex, ey, z_lift]))

# 添加最终停靠点（可选）
final_q = np.array([0.0, -2.32, -1.38, -2.45, 1.57, 0.0])

# 时间参数
dt_control = 0.02  # 与 data.time += 0.02 一致
speed = 0.05        # m/s，末端移动速度
current_traj_index = 0
t_start_segment = 0.0

# 预计算每段所需时间
segment_times = []
segment_points = []

for i in range(len(trajectory) - 1):
    p0 = trajectory[i]
    p1 = trajectory[i+1]
    dist = np.linalg.norm(p1 - p0)
    t_seg = dist / speed if dist > 1e-6 else 0.1  # 至少 0.1s 避免除零
    segment_times.append(t_seg)
    segment_points.append((p0, p1))

total_segments = len(segment_points)

point_A = np.array([0.3, 0.1, 0.1])
point_B = np.array([0.6, 0.6, 0.1])
duration_A = 1
duration_B = 200
final_q = np.array([0.0,-2.32,-1.38,-2.45,1.57,0.0])
######################################
## USER CODE ENDS HERE
######################################

while not glfw.window_should_close(window):
    time_prev = data.time

    while (data.time - time_prev < 0.1):
        # Store trajectory
        mj_end_eff_pos = data.site_xpos[0]
        # Pen-down detection: allow small tolerance around z_write to avoid false breaks
        # Use <= z_write + tol instead of hardcoded 0.1 and strict <
        tol = 1e-5
        if (mj_end_eff_pos[2] <= z_write + tol):
            traj_points.append(mj_end_eff_pos.copy())
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)
            
        # Get current joint configuration
        cur_q_pos = data.qpos.copy()
        
        ######################################
        ## USER CODE STARTS HERE
        ######################################
        if current_traj_index >= total_segments:
            X_ref = np.array([0.0, 0.0, 0.2])  # 悬停
        else:
            p0, p1 = segment_points[current_traj_index]
            t_elapsed = data.time- t_start_segment
            t_seg = segment_times[current_traj_index]

            if t_elapsed >= t_seg:
                # 进入下一段
                current_traj_index += 1
                t_start_segment = data.time
                if current_traj_index < total_segments:
                    p0, p1 = segment_points[current_traj_index]
                    t_elapsed = 0.0
                    t_seg = segment_times[current_traj_index]

            # 线性插值
            alpha = min(1.0, t_elapsed / t_seg) if t_seg > 0 else 1.0
            X_ref = (1 - alpha) * p0 + alpha * p1
        
        ######################################
        ## USER CODE ENDS HERE
        ######################################

        # Compute control input using IK
        cur_ctrl = IK_controller(model, data, X_ref, cur_q_pos)
        
        # Apply control input
        if data.time < duration_B:
            data.ctrl[:] = cur_ctrl
        else:
            data.ctrl[:] = final_q
        mj.mj_step(model, data)
        data.time += 0.01 #datatime是模拟时间，需要手动更新，与控制频率对应，用来控制模拟速度

    if (data.time>=simend):
        break

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth = ',cam.azimuth,'\n','cam.elevation = ',cam.elevation,'\n','cam.distance = ',cam.distance)
        print('cam.lookat = np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    # Add trajectory as spheres
    for j in range(1, len(traj_points)):
        if scene.ngeom >= scene.maxgeom:
            break  # avoid overflow

        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        
        # Configure this geom as a line
        geom.type = mj.mjtGeom.mjGEOM_SPHERE  # Use sphere for endpoints
        geom.rgba[:] = LINE_RGBA
        geom.size[:] = np.array([0.002, 0.002, 0.002])
        geom.pos[:] = traj_points[j]
        geom.mat[:] = np.eye(3)  # no rotation
        geom.dataid = -1
        geom.segid = -1
        geom.objtype = 0
        geom.objid = 0
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
