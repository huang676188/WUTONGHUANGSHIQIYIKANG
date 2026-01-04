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
    Jinv = np.linalg.pinv(J)#计算雅可比矩阵的伪逆

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
MAX_TRAJ = 5e6  # Maximum number of trajectory points to store
LINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0])

######################################
## USER CODE STARTS HERE
######################################
strokes = [
[np.array([-0.38,0.55]),np.array([-0.35,0.44]),np.array([-0.31,0.55]),np.array([-0.27,0.43]),np.array([-0.24,0.55])],
[np.array([-0.21,0.54]),np.array([-0.21,0.46]),np.array([-0.18,0.43]),np.array([-0.16,0.43]),np.array([-0.13,0.46]),np.array([-0.13,0.55])],
[np.array([-0.1,0.54]),np.array([-0.01,0.54])],
[np.array([-0.06,0.54]),np.array([-0.05,0.43])],
[np.array([0.06,0.54]),np.array([0.02,0.53]),np.array([0.01,0.48]),np.array([0.03,0.44]),np.array([0.07,0.43]),np.array([0.1,0.45]),np.array([0.11,0.5]),np.array([0.09,0.53]),np.array([0.06,0.54])],
[np.array([0.15,0.43]),np.array([0.15,0.54]),np.array([0.23,0.44]),np.array([0.23,0.54])],
[np.array([0.36,0.52]),np.array([0.32,0.55]),np.array([0.27,0.52]),np.array([0.27,0.47]),np.array([0.31,0.43]),np.array([0.36,0.45])],
[np.array([0.32,0.49]),np.array([0.36,0.48]),np.array([0.36,0.43])],
[np.array([-0.2368, 0.2998]), np.array([-0.2286, 0.2902]), np.array([-0.221, 0.2578])],
[np.array([-0.2302, 0.301]), np.array([-0.2292, 0.2996]), np.array([-0.2214, 0.2976]), np.array([-0.1812, 0.3064]), np.array([-0.1718, 0.306]), np.array([-0.165, 0.2996]), np.array([-0.1716, 0.2808]), np.array([-0.1768, 0.2788])],
[np.array([-0.2168, 0.2616]), np.array([-0.2148, 0.2644]), np.array([-0.1802, 0.2712]), np.array([-0.1698, 0.2716]), np.array([-0.165, 0.2704])],
[np.array([-0.2378, 0.2378]), np.array([-0.23, 0.2362]), np.array([-0.2218, 0.2366]), np.array([-0.1656, 0.2474]), np.array([-0.1562, 0.2472])],
[np.array([-0.2682, 0.2022]), np.array([-0.2572, 0.2002]), np.array([-0.2274, 0.2052]), np.array([-0.1436, 0.214]), np.array([-0.1334, 0.2124]), np.array([-0.126, 0.2098])],
[np.array([-0.2086, 0.2328]), np.array([-0.2008, 0.2276]), np.array([-0.2008, 0.224]), np.array([-0.2042, 0.2034]), np.array([-0.2126, 0.1832]), np.array([-0.2182, 0.1754]), np.array([-0.2262, 0.1674]), np.array([-0.2408, 0.1588]), np.array([-0.2556, 0.153]), np.array([-0.2608, 0.1522])],
[np.array([-0.197, 0.2032]), np.array([-0.1946, 0.1984]), np.array([-0.1926, 0.1974]), np.array([-0.1818, 0.1844]), np.array([-0.1684, 0.171]), np.array([-0.1546, 0.1596]), np.array([-0.1428, 0.1554]), np.array([-0.1154, 0.1502])],
[np.array([-0.0552, 0.2998]), np.array([-0.0486, 0.2926]), np.array([-0.0468, 0.2868]), np.array([-0.0458, 0.2542]), np.array([-0.0502, 0.1956]), np.array([-0.0544, 0.1756]), np.array([-0.0546, 0.1626])],
[np.array([-0.0416, 0.2958]), np.array([-0.037, 0.2926]), np.array([0.0416, 0.3068]), np.array([0.0488, 0.3042]), np.array([0.0544, 0.2984]), np.array([0.054, 0.2864]), np.array([0.0582, 0.192]), np.array([0.0576, 0.1684]), np.array([0.0542, 0.16]), np.array([0.0526, 0.16]), np.array([0.0426, 0.1618]), np.array([0.025, 0.1684])],
[np.array([-0.0246, 0.2654]), np.array([-0.0226, 0.2644]), np.array([-0.0122, 0.2642]), np.array([0.015, 0.2706]), np.array([0.0256, 0.2708])],
[np.array([-0.028, 0.2386]), np.array([-0.023, 0.2342]), np.array([-0.0218, 0.2312]), np.array([-0.0158, 0.1986])],
[np.array([-0.0168, 0.234]), np.array([-0.0134, 0.2366]), np.array([0.0082, 0.2418]), np.array([0.0124, 0.2426]), np.array([0.0172, 0.241]), np.array([0.0216, 0.2366]), np.array([0.0166, 0.2198]), np.array([0.0124, 0.2166])],
[np.array([-0.012, 0.206]), np.array([-0.0092, 0.2082]), np.array([0.0106, 0.2116]), np.array([0.0182, 0.2116]), np.array([0.023, 0.21])],
[np.array([0.2528, 0.2788]), np.array([0.2632, 0.2766]), np.array([0.3396, 0.2874]), np.array([0.3478, 0.2872]), np.array([0.3526, 0.2858])],
[np.array([0.2716, 0.3044]), np.array([0.28, 0.2962]), np.array([0.2804, 0.2944]), np.array([0.2844, 0.2546]), np.array([0.2818, 0.2512])],
[np.array([0.3202, 0.3138]), np.array([0.3242, 0.31]), np.array([0.3274, 0.3044]), np.array([0.3182, 0.258]), np.array([0.3142, 0.2542])],
[np.array([0.2214, 0.2408]), np.array([0.2328, 0.2386]), np.array([0.288, 0.2464]), np.array([0.3644, 0.253]), np.array([0.3752, 0.2516]), np.array([0.3832, 0.2484])],
[np.array([0.2548, 0.2256]), np.array([0.2612, 0.2202]), np.array([0.2718, 0.177])],
[np.array([0.2668, 0.2244]), np.array([0.27, 0.2224]), np.array([0.3258, 0.2306]), np.array([0.3372, 0.2294]), np.array([0.3434, 0.2218]), np.array([0.3386, 0.2098]), np.array([0.3302, 0.1812])],
[np.array([0.2792, 0.2042]), np.array([0.3102, 0.2086]), np.array([0.3206, 0.2076])],
[np.array([0.2936, 0.2418]), np.array([0.2988, 0.2388]), np.array([0.3008, 0.2336]), np.array([0.3002, 0.1916]), np.array([0.297, 0.1874])],
#发生什么事了
[np.array([0.2756, 0.1788]), np.array([0.2794, 0.1808]), np.array([0.3216, 0.1858]), np.array([0.3254, 0.1872])],
[np.array([0.2878, 0.1648]), np.array([0.2802, 0.1648]), np.array([0.2718, 0.1572]), np.array([0.2594, 0.149]), np.array([0.24, 0.141])],
[np.array([0.3196, 0.1702]), np.array([0.343, 0.1552]), np.array([0.3494, 0.148]), np.array([0.3522, 0.1416])]

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
#这块代码的作用是将书写轨迹分割成多个线段，并计算每段所需的时间

total_segments = len(segment_points) #计算总段数，用于后续判断是否完成书写

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
    count = 0
    while (data.time - time_prev < 5):
        # Store trajectory
        mj_end_eff_pos = data.site_xpos[0]
        # Pen-down detection: allow small tolerance around z_write to avoid false breaks
        # Use <= z_write + tol instead of hardcoded 0.1 and strict <
        tol = 1e-5
        count += 1
        if (mj_end_eff_pos[2] <= z_write + tol and count % 2 == 0):
            traj_points.append(mj_end_eff_pos.copy())
            count = 0
        if len(traj_points) > MAX_TRAJ:
            traj_points.pop(0)
            
        # Get current joint configuration
        cur_q_pos = data.qpos.copy()
        
        ######################################
        ## USER CODE STARTS HERE
        ######################################
        if np.all(p0 == np.array([0.2936, 0.2418, 0.1])):
            print("到达倒数第四笔")
        if np.all(p0 == np.array([0.2756, 0.1788, 0.1])):
            print("到达倒数第三笔")
        if np.all(p0 == np.array([0.2878, 0.1648, 0.1])):
            print("到达倒数第二笔")
        if np.all(p0 == np.array([0.3196, 0.1702, 0.1])):
            print("到达最后一笔")
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
        if scene.ngeom >= scene.maxgeom:   #这个值在最开始定义为10000，用来限制最大显示的轨迹点数
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
