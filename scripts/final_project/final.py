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
[np.array([-0.06,0.54]),np.array([-0.06,0.43])],
[np.array([0.06,0.54]),np.array([0.02,0.53]),np.array([0.01,0.48]),np.array([0.03,0.44]),np.array([0.07,0.43]),np.array([0.1,0.45]),np.array([0.11,0.5]),np.array([0.09,0.53]),np.array([0.06,0.54])],
[np.array([0.15,0.43]),np.array([0.15,0.54]),np.array([0.23,0.43]),np.array([0.23,0.54])],
[np.array([0.36,0.52]),np.array([0.32,0.55]),np.array([0.27,0.52]),np.array([0.27,0.47]),np.array([0.31,0.43]),np.array([0.36,0.45])],
[np.array([0.32,0.48]),np.array([0.36,0.48]),np.array([0.36,0.43])],
[np.array([-0.37,0.33]),np.array([-0.35,0.35]),np.array([-0.33,0.35]),np.array([-0.33,0.34]),np.array([-0.33,0.32]),np.array([-0.37,0.28]),np.array([-0.38,0.27]),np.array([-0.33,0.27])],
[np.array([-0.28,0.35]),np.array([-0.3,0.34]),np.array([-0.3,0.29]),np.array([-0.3,0.27]),np.array([-0.28,0.27]),np.array([-0.26,0.29]),np.array([-0.25,0.33]),np.array([-0.26,0.35]),np.array([-0.28,0.35])],
[np.array([-0.22,0.33]),np.array([-0.21,0.35]),np.array([-0.19,0.35]),np.array([-0.18,0.33]),np.array([-0.17,0.32]),np.array([-0.22,0.29]),np.array([-0.23,0.27]),np.array([-0.19,0.27])],
[np.array([-0.1,0.35]),np.array([-0.14,0.35]),np.array([-0.15,0.31]),np.array([-0.13,0.32]),np.array([-0.11,0.31]),np.array([-0.11,0.29]),np.array([-0.13,0.27]),np.array([-0.15,0.27]),np.array([-0.16,0.29]),],
[np.array([-0.03,0.35]),np.array([-0.07,0.35]),np.array([-0.08,0.32]),np.array([-0.06,0.32]),np.array([-0.04,0.32]),np.array([-0.04,0.29]),np.array([-0.05,0.27]),np.array([-0.07,0.27]),np.array([-0.09,0.29]),],
[np.array([-0.37,0.19]),np.array([-0.35,0.21]),np.array([-0.33,0.2]),np.array([-0.32,0.19]),np.array([-0.34,0.17]),np.array([-0.35,0.17]),np.array([-0.33,0.16]),np.array([-0.33,0.14]),np.array([-0.35,0.12]),np.array([-0.37,0.13]),np.array([-0.38,0.14])],
[np.array([-0.26,0.21]),np.array([-0.28,0.12])],
[np.array([-0.19,0.21]),np.array([-0.21,0.12])],
[np.array([-0.12,0.21]),np.array([-0.14,0.12])],
[np.array([-0.06,0.21]),np.array([-0.08,0.19]),np.array([-0.09,0.15]),np.array([-0.08,0.13]),np.array([-0.06,0.13]),np.array([-0.04,0.15]),np.array([-0.03,0.19]),np.array([-0.04,0.21]),np.array([-0.06,0.21])],
[np.array([0.0132, 0.3998]), np.array([0.0214, 0.3902]), np.array([0.029, 0.3578])],
[np.array([0.0198, 0.401]), np.array([0.0208, 0.3996]), np.array([0.0286, 0.3976]), np.array([0.0688, 0.4064]), np.array([0.0782, 0.406]), np.array([0.085, 0.3996]), np.array([0.0784, 0.3808]), np.array([0.0732, 0.3788])],
[np.array([0.0332, 0.3616]), np.array([0.0352, 0.3644]), np.array([0.0698, 0.3712]), np.array([0.0802, 0.3716]), np.array([0.085, 0.3704])],
[np.array([0.0122, 0.3378]), np.array([0.02, 0.3362]), np.array([0.0282, 0.3366]), np.array([0.0844, 0.3474]), np.array([0.0938, 0.3472])],
[np.array([-0.0182, 0.3022]), np.array([-0.0072, 0.3002]), np.array([0.0226, 0.3052]), np.array([0.1064, 0.314]), np.array([0.1166, 0.3124]), np.array([0.124, 0.3098])],
[np.array([0.0414, 0.3328]), np.array([0.0492, 0.3276]), np.array([0.0492, 0.324]), np.array([0.0458, 0.3034]), np.array([0.0374, 0.2832]), np.array([0.0318, 0.2754]), np.array([0.0238, 0.2674]), np.array([0.0092, 0.2588]), np.array([-0.0056, 0.253]), np.array([-0.0108, 0.2522])],
[np.array([0.053, 0.3032]), np.array([0.0554, 0.2984]), np.array([0.0574, 0.2974]), np.array([0.0682, 0.2844]), np.array([0.0816, 0.271]), np.array([0.0954, 0.2596]), np.array([0.1072, 0.2554]), np.array([0.1346, 0.2502])],
[np.array([0.1748, 0.3998]), np.array([0.1814, 0.3926]), np.array([0.1832, 0.3868]), np.array([0.1842, 0.3542]), np.array([0.1798, 0.2956]), np.array([0.1756, 0.2756]), np.array([0.1754, 0.2626])],
[np.array([0.1884, 0.3958]), np.array([0.193, 0.3926]), np.array([0.2716, 0.4068]), np.array([0.2788, 0.4042]), np.array([0.2844, 0.3984]), np.array([0.284, 0.3864]), np.array([0.2882, 0.292]), np.array([0.2876, 0.2684]), np.array([0.2842, 0.26]), np.array([0.2826, 0.26]), np.array([0.2726, 0.2618]), np.array([0.255, 0.2684])],
[np.array([0.2054, 0.3654]), np.array([0.2074, 0.3644]), np.array([0.2178, 0.3642]), np.array([0.245, 0.3706]), np.array([0.2556, 0.3708])],
[np.array([0.202, 0.3386]), np.array([0.207, 0.3342]), np.array([0.2082, 0.3312]), np.array([0.2142, 0.2986])],
[np.array([0.2132, 0.334]), np.array([0.2166, 0.3366]), np.array([0.2382, 0.3418]), np.array([0.2424, 0.3426]), np.array([0.2472, 0.341]), np.array([0.2516, 0.3366]), np.array([0.2466, 0.3198]), np.array([0.2424, 0.3166])],
[np.array([0.218, 0.306]), np.array([0.2208, 0.3082]), np.array([0.2406, 0.3116]), np.array([0.2482, 0.3116]), np.array([0.253, 0.31])],
[np.array([0.3528, 0.2788]), np.array([0.3632, 0.2766]), np.array([0.4396, 0.2874]), np.array([0.4478, 0.2872]), np.array([0.4526, 0.2858])],
[np.array([0.3716, 0.3044]), np.array([0.38, 0.2962]), np.array([0.3804, 0.2944]), np.array([0.3844, 0.2546]), np.array([0.3818, 0.2512])],
[np.array([0.4202, 0.3138]), np.array([0.4242, 0.31]), np.array([0.4274, 0.3044]), np.array([0.4182, 0.258]), np.array([0.4142, 0.2542])],
[np.array([0.3214, 0.2408]), np.array([0.3328, 0.2386]), np.array([0.388, 0.2464]), np.array([0.4644, 0.253]), np.array([0.4752, 0.2516]), np.array([0.4832, 0.2484])],
[np.array([0.3548, 0.2256]), np.array([0.3612, 0.2202]), np.array([0.3718, 0.177])],
[np.array([0.3668, 0.2244]), np.array([0.37, 0.2224]), np.array([0.4258, 0.2306]), np.array([0.4372, 0.2294]), np.array([0.4434, 0.2218]), np.array([0.4386, 0.2098]), np.array([0.4302, 0.1812])],
[np.array([0.3792, 0.2042]), np.array([0.4102, 0.2086]), np.array([0.4206, 0.2076])],
[np.array([0.3936, 0.2418]), np.array([0.3988, 0.2388]), np.array([0.4008, 0.2336]), np.array([0.4002, 0.1916]), np.array([0.397, 0.1874])],
[np.array([0.3756, 0.1788]), np.array([0.3794, 0.1808]), np.array([0.4216, 0.1858]), np.array([0.4254, 0.1872])],
[np.array([0.3878, 0.1648]), np.array([0.3802, 0.1648]), np.array([0.3718, 0.1572]), np.array([0.3594, 0.149]), np.array([0.34, 0.141])],
[np.array([0.4196, 0.1702]), np.array([0.443, 0.1552]), np.array([0.4494, 0.148]), np.array([0.4522, 0.1416])]
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
    segment_times.append(t_seg/1.1)
    segment_points.append((p0, p1))
#这块代码的作用是将书写轨迹分割成多个线段，并计算每段所需的时间

total_segments = len(segment_points) #计算总段数，用于后续判断是否完成书写

point_A = np.array([0.3, 0.1, 0.1])
point_B = np.array([0.6, 0.6, 0.1])
duration_A = 1
duration_B = sum(segment_times) + 5.0
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
        mj.mj_step(model, data) #mj_step用来推进模拟一步，意思就是让模拟时间往前走一点点
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
