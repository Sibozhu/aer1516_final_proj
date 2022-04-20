from importlib.resources import path
from telnetlib import DO
# from black import first_leaf_column
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch import argmin
import bisect

def circle_intersect(x1,x2,xc,rc,R):
    d3 = np.linalg.norm(x2 - x1)
    d1 = np.linalg.norm(x1 - xc)
    d2 = np.linalg.norm(x2 - xc)
    dmax = max(d1,d2)
    dmin = min(d1,d2)
    if dmax >= d3:
        c_theta = (d3**2 + dmin**2 - dmax**2)/(2*dmin*d3)
        if c_theta >= 0:
            a = -c_theta*dmin
            r = np.sqrt(dmin**2-a**2)
        else:
            r = dmin
    else:            
        a = (d3**2 + dmin**2 - dmax**2)/(2*d3)
        r = np.sqrt(dmin**2-a**2)
    if r <= (rc + 2*R):
        return True
    else:
        return False 


class planner():
    def __init__(self, robots, radius, vmax):
        self.t_curr = 0
        self.t_next = 0
        self.robots = robots
        self.ids_OMP = {robot.id for robot in robots}
        self.ids_CHOP = {}
        self.CHOPs = []
        self.vmax = vmax
        self.R = radius
        # self.new_OMP = {robot.idx for robot in robots}
    
    def update_state(self, t):
        self.t_curr = t
        for robot in self.robots:
            idx = bisect.bisect_left(robot.t_list, t)
            if dt > robot.dt[idx]:
                robot.x_curr = robot.goals[idx]
                robot.goal_curr = robot.goals[idx]
                robot.stop = True
            elif idx:
                dir = (robot.goals[idx] - robot.goals[idx-1])
                self.x_curr = robot.goals[idx-1] + dir/np.linalg.norm(dir)
                self.goal_curr = robot.goal[idx]

    def find_imminent_collision(self, t):
        # TO-DO
        return None

    def check_collision(self):
        # self.update_states()
        
        n = len(self.robots)
        self.collision_times = np.ones((2,n,n)) * 1000000
        self.ids_CHOP
        ids_intersect = None
        ts_intersect = None
        for i in range(n):
            xi = self.robots[i].x_curr
            gi = self.robots[i].goal_curr
            vect_i = gi - xi
            nvi = vect_i / np.linalg.norm(vect_i)
            for j in range(i+1, len(self.robots)):
                xj = self.robots[j].x_curr
                gj = self.robots[j].goal_curr
                vect_j = gj - xj
                nvj = vect_j / np.linalg.norm(vect_j)
                dnv = nvi - nvj
                dx = xi - xj
                a = dnv @ dnv
                b = 2 * dnv @ dx
                c = dx @ dx - 8 * self.robots[i].R**2
                val = b**2 - 4*a*c
                if val > 0:
                    t1 = (-b + np.sqrt(val)) / 2 / a
                    t2 = (-b - np.sqrt(val)) / 2 / a
                    self.collision_times[0,i,j] = min(t1, t2)
                    self.collision_times[1,i,j] = max(t1, t2)
                else:
                    if c > 0:
                        self.collision_times[0,i,j] = 1000000
                        self.collision_times[1,i,j] = 1000000
                    else:
                        print('Robot {} and {} always intersect'.format(i,j))

        ids_intersect = np.unravel_index(self.collision_times.argmin(), self.collision_times.shape)
        ts_intersect = self.collision_times[ids_intersect]
        ids_intersect = ids_intersect[1:]

        if ts_intersect is not None:
            robots_in = [self.robots[x] for x in ids_intersect]

        while ts_intersect is not None:
            print(ts_intersect)
            Madd = []
            Mcurr = []
            Romp = []
            Radd = robots_in
            ts = ts_intersect
            # print(ts)
            print('######')
            merge = 1
            for chop in self.CHOPs:
                t_start = chop.t_crit[0]
                t_end = chop.t_crit[-1]
                if [rob for rob in chop.robots if (rob in Radd) and (ts > t_start) and (ts < t_end)]:
                    if chop not in Mcurr:
                        Mcurr.append(chop)
            
            while True:
                n = len(Radd)
                starts = np.zeros((n,2))
                goals = np.zeros((n,2))
                for i in range(n):
                    dir = Radd[i].goal_curr - Radd[i].x_curr
                    starts[i,:] = Radd[i].x_curr + dir / np.linalg.norm(dir) * (ts - Radd[i].t_curr) * self.vmax
                    goals[i,:] = Radd[i].goal_curr
                if not merge:
                    break
                
                new_chop = CHOP(ts, Radd, starts, goals, self.R, self.vmax)
                new_chop.build_CHOP()

                ta = new_chop.ts
                tb = new_chop.t_crit[-1]
                ts_min = [new_chop.ts for robot in Radd]
                for robot in self.robots:
                    if robot in Radd:
                        continue
                    # ids = [i for i in range(len(robot.ts)) if (robot.ts[i] >= ta and robot.ts[i]<= tb)]
                    id1 = bisect.bisect_right(robot.ts, ta)-1
                    id2 = bisect.bisect_right(robot.ts, tb)
                    for i in range(id1,id2):
                        t1 = robot.ts[id1]
                        t2 = robot.ts[id1+1]
                        x1 = robot.path[id1]
                        x2 = robot.path[id1+1]
                        if t1 < ta and t2 > tb:
                            vec = x2 - x1
                            nvec = vec / np.linalg.norm(vec)
                            x2 = x1 + nvec * (tb-t1) * self.vmax
                            x1 = x1 + nvec * (ta-t1) * self.vmax
                        elif t1 < ta:
                            vec = x2 - x1
                            nvec = vec / np.linalg.norm(vec)
                            x1 = x1 + nvec * (ta-t1) * self.vmax
                        elif t2 > tb:
                            vec = x2 - x1
                            nvec = vec / np.linalg.norm(vec)
                            x2 = x1 + nvec * (tb-t1) * self.vmax
                        if circle_intersect(x1,x2,new_chop.xc,new_chop.r,self.R):
                            if robot.in_CHOP[i]:
                                min_ts = min(new_chop.ts, robot.in_CHOP[i].ts)
                                ts_min = ts_min + [min_ts for rob in robot.in_CHOP[i+1].robots if rob not in Radd]
                                Radd = Radd + [rob for rob in robot.in_CHOP[i].robots if rob not in Radd]
                                Mcurr.append(robot.in_CHOP[i])
                            # elif robot.in_CHOP[i+1]:
                            #     min_ts = min(new_chop.ts, robot.in_CHOP[i+1].ts)
                            #     ts_min = ts_min + [min_ts for rob in robot.in_CHOP[i+1].robots if rob not in Radd]
                            #     Radd = Radd + [rob for rob in robot.in_CHOP[i+1].robots if rob not in Radd]
                            #     Mcurr.append(robot.in_CHOP[i+1])
                            else:
                                Radd.append(robot)
                                Romp.append(robot)
                                ts_min.append(new_chop.ts)

                for chop in self.CHOPs:
                    if chop not in Mcurr:
                        for vals in enumerate(chop.robots):
                            t1 = chop.t_lists[vals[0]][-1]
                            t2 = [ts_min[x[0]] for x in enumerate(Radd) if x[0] in chop.robots]
                            n_shared = sum([1 for x in Radd if x in chop.robots])
                            if t1 >= t1 or n_shared >= 2:
                                if chop not in Mcurr:
                                    Mcurr.append(chop)

                for chop in Mcurr:
                    Radd = Radd + [rob for rob in chop.robots if rob not in Radd]
                Madd = Madd + [chop for chop in Mcurr if chop not in Madd]
                ts_all = [chop.ts for chop in Mcurr]
                n_added_chop = sum([1 for chop in Mcurr if chop not in Madd])
                if Romp or n_added_chop:
                    merge = 1
                    if ts_all:
                        ts = min(new_chop.ts, *ts_all)
                    else:
                        ts = new_chop.ts      
                    Romp = []
                    Mcurr = []
                else:
                    merge = 0

                # print(f'ts:{ts}')
                # uuuu

            new_chop = CHOP(ts, Radd, starts, goals, self.R, self.vmax)
            new_chop.build_CHOP()
            self.CHOPs.append(new_chop)


            for chop in Madd:
                
                # print(f'######')
                for robot in chop.robots:

                    ids = [x[0] for x in enumerate(robot.in_CHOP) if x[1] == chop]
                    # print(f'ids: {ids}')
                
                    ids_flag = [ids[0], ids[-1]]

                    del robot.in_CHOP[ids_flag[0]:ids_flag[-1]+1]
                    del robot.path[ids_flag[0]:ids_flag[-1]+1]
                    del robot.ts[ids_flag[0]:ids_flag[-1]+1]

                self.CHOPs.remove(chop)
            print(self.CHOPs[0].robots[0].path)
            print('^^^^^^^^^^^^')
            # www
            ts_merged = []
            
            for robot in new_chop.robots:
                idx = [x[0] for x in enumerate(new_chop.robots) if x[1] == robot][0]
                id_insert = bisect.bisect_right(robot.ts, new_chop.ts)
                robot.ts = robot.ts[:id_insert] + new_chop.t_lists[idx]
                ts_merged += robot.ts

                robot.in_CHOP = robot.in_CHOP[:id_insert] + [new_chop for x in range(len(new_chop.t_lists[idx]))]
                robot.in_CHOP[-1] = 0
                robot.path = robot.path[:id_insert] + new_chop.goals[idx]
                # t_list = robot.ts
                # goal_list = robot.path
                # in_chop = robot.in_CHOP
                # for chop in self.CHOPs:
                #     if robot in chop.robots:
                #         i = chop.robots.index(robot)
                #         t_list.append(chop.t_lists[i])
                #         goal_list.append(chop.goals[i])
                #         in_chop = in_chop + [True for x in range(len(chop.goals[i]))]
                # test = 1
            # self.find_imminent_collision # like line 69
            # #check starting point / end point 
            print(new_chop.robots[0].ts,len(new_chop.robots[0].ts))
            print(new_chop.robots[0].path,len(new_chop.robots[0].path))
            print('&&&&&&&&&')
            print(new_chop.robots[1].ts,len(new_chop.robots[1].ts))
            print(new_chop.robots[1].path, len(new_chop.robots[1].path))
            print('@@@@@@@@@')
            print(new_chop.robots[2].ts,len(new_chop.robots[2].ts))
            print(new_chop.robots[2].path, len(new_chop.robots[2].path))

            ts_merged = np.unique(np.sort(ts_merged))
            wwwwwww
            
            self.collition_times = ones(2,n,n)*100000
            for frame_id in len(ts_merged):
                for i in range(n):
                    xi = self.robots[i].x_curr
                    gi = self.robots[i].goal_curr
                    vect_i = gi - xi
                    nvi = vect_i / np.linalg.norm(vect_i) 
                    for j in range(i+1, len(self.robots)):

                        xj = self.robots[j].x_curr
                        gj = self.robots[j].goal_curr
                        vect_j = gj - xj
                        nvj = vect_j / np.linalg.norm(vect_j)

                        dnv = (nvi - nvj) * self.vmax
                        dx = xi - xj
                        a = dnv @ dnv
                        b = 2 * dnv @ dx
                        c = dx @ dx - 8 * self.robots[i].R**2
                        val = b**2 - 4*a*c
                        if val > 0:
                            t1 = (-b + np.sqrt(val)) / 2 / a
                            t2 = (-b - np.sqrt(val)) / 2 / a
                            if t1 < ts_merged[frame_id+1] and t1 > ts_merged[frame_id]: # time < t1 < time+1:
                                if t1 < self.collision_times[0,i,j]:
                                    self.collision_times[0,i,j] = t1
                            elif t2 < ts_merged[frame_id+1]:
                                if t2 < self.collision_times[0,i,j]:
                                    self.collision_times[0,i,j] = t2
                            else: 
                                pass

                        else:
                            if c > 0: # they will never collide
                                self.collision_times[0,i,j] = 1000000
                                self.collision_times[1,i,j] = 1000000
                            else: # they are parallel and always intersect
                                print('Robot {} and {} always intersect'.format(i,j))

            ids_intersect = np.unravel_index(self.collision_times.argmin(), self.collision_times.shape)
            ts_intersect = self.collision_times[ids_intersect]
            
            if min(ts_intersect) == 1000000:
                ts_intersect = None
            # print(ts_intersect_flag)
            # print('$$$$$$$$')

        

    def timestep(self, t):
        pass
    
    def enter_CHOP(self, robots, ):
        pass


class CHOP():
    def __init__(self, start_time, robots, starts, goals, radius, vmax):
        self.ts = start_time
        self.robots = robots
        self.Xs = starts
        self.Xg = goals
        self.R = radius
        self.vmax = vmax
        self.Nm = len(self.robots)
        self.xc = np.average(starts,axis=0)
        self.nw = 2*self.Nm
        self.curr_pos = []
        self.in_CHOP = False
        self.t_crit = [start_time]

    def build_CHOP(self):
        min_dist_node = 2*np.sqrt(2)*self.R
        min_dist_line = 2*self.R
        path_r = min_dist_node / 2 / np.sin(np.pi/self.nw)
        angs = np.linspace(0,1-1/self.nw, self.nw) * np.pi * 2
        cs = np.cos(angs)
        ss = np.sin(angs)
        dist_to_xc = np.sqrt((self.Xg[:,0]-self.xc[0])**2 + (self.Xg[:,1]-self.xc[1])**2)
        big_R = path_r + 2*np.sqrt(2)*self.R
        small_R = min(path_r - 2*np.sqrt(2)*self.R, path_r*np.cos(angs[1]/2) - 2*self.R)
        if any((dist_to_xc < big_R) * (dist_to_xc > small_R)):
            max_r = max(dist_to_xc)
            path_r = max(max_r + 2*np.sqrt(2)*self.R, (max_r + 2*self.R)/np.cos(angs[1]/2))
        points = np.ones((self.nw, 2)) * path_r
        points[:,0] *= cs
        points[:,1] *= ss
        points += self.xc

        self.r = path_r
        self.Xm = points
        self.Xw = points[::2]
        cost_mat = np.zeros((self.Nm,self.Nm))
        xw2 = np.zeros(self.Nm)
        for i in range(self.Nm):
            cost_mat[i,:] = (self.Xw[:,0]-self.Xs[i,0])**2 + (self.Xw[:,1]-self.Xs[i,1])**2
            gx = self.Xg[i,0]
            gy = self.Xg[i,1]
            dists = (points[:,0]-gx)**2 + (points[:,1]-gy)**2
            xw2[i] = np.argmin(dists)
        xs, xw1 = linear_sum_assignment(cost_mat)

        self.matches = np.vstack((xs,xw1,xw2))
        self.matches = self.matches[:,self.matches[0,:].argsort()].astype(int)
        self.priority = [[] for i in range(self.Nm)]
        for i in range(self.Nm):
            id_exit = self.matches[2,i]
            exit_traj = self.Xg[i,:] - points[id_exit,:]
            d3 = np.sqrt(exit_traj[0]**2+exit_traj[1]**2)
            d1s = np.linalg.norm(self.Xg - self.Xg[i,:], axis=1)
            d2s = np.linalg.norm(self.Xg - points[id_exit,:], axis=1)
            for j in range(self.Nm):
                if j == i:
                    continue
                d1 = d1s[j]
                d2 = d2s[j]
                alpha = (d1**2-d2**2+d3**2) / (2*d3)
                if (d3 - alpha) > d3 or alpha > d3:
                    dist = min(d1,d2)
                else:
                    dist = np.sqrt(d1**2 - alpha**2)
                if dist < 2*self.R:
                    self.priority[i].append(j)
        
        self.curr_step = 0
        self.t_lists = [[self.ts] for robot in self.robots]
        self.t_exit = [[] for robot in self.robots]
        dist_max = 0
        id_max = None
        self.goals = [[] for robot in self.robots]
        self.goal_id = [0 for robot in self.robots]
        for i in range(self.Nm):
            id_first = self.matches[1,i]
            dist = np.linalg.norm(self.Xs[i,:] - self.Xw[id_first, :])
            self.goals[i].append(self.Xw[id_first,:])
            self.goal_id[i] = id_first * 2
            self.t_lists[i].append(dist/self.vmax + self.t_lists[0][0])
            if dist > dist_max:
                dist_max = dist
                id_max = i
        t1 = dist_max/self.vmax + self.t_lists[0][0]
        for i in range(self.Nm):
            if not id_max:
                self.t_lists[i].append(t1)
        self.robot_out = []
        t_step = np.linalg.norm(self.Xm[0, :]-self.Xm[1, :]) / self.vmax
        is_done = False
        while True:
            for i in range(self.Nm):
                exit_id = self.matches[2,i]
                if all(self.goals[i][-1] == self.Xm[exit_id, :]) and all(x in self.robot_out for x in self.priority[i]):
                    self.goals[i].append(self.Xg[i,:])
                    self.goal_id[i] = None
                    self.robot_out.append(i)
                    dt = np.linalg.norm(self.goals[i][-1]-self.goals[i][-2]) / self.vmax
                    self.t_exit[i] = self.t_lists[i][-1]
                    self.t_crit.append(self.t_lists[i][-1])
                    self.t_lists[i].append(self.t_lists[i][-1] + dt)
                    if all([x is None for x in self.goal_id]):
                        self.t_chop = self.t_lists[i][:-1]
                        is_done = True
                else:
                    if self.goal_id[i] is not None:
                        if not self.goal_id[i]:
                            self.goal_id[i] = self.nw - 1
                        else:
                            self.goal_id[i] -= 1
                        self.goals[i].append(self.Xm[self.goal_id[i],:])
                        self.t_lists[i].append(self.t_lists[i][-1] + t_step)
                        # print(i,self.goal_id[i])
                if is_done:
                    break
            if is_done:
                break
        # for i in range(self.Nm):
        #     self.robots[i].update_path(self.t_lists[i], self.goals[i])


class new_robot():
    def __init__(self, idx, start, goal, vmax, radius):
        self.id = idx
        self.start = start
        self.goal = goal
        self.path = [start, goal]
        tf = np.linalg.norm(goal-start)/ vmax
        self.ts = [0, tf]
        self.t_curr = 0
        self.x_curr = start
        self.goal_curr = goal
        self.vmax = vmax
        self.in_CHOP = [0,0]
        self.R = radius

    def timestep(self, t):
        
        dir = self.goal_curr - self.pos[-1]
        dir = dir / np.linalg.norm(dir)
        new_pos = self.pos[-1] + (t - self.f) * dir * self.vmax
        self.pos.append(new_pos)

    def update_path(self, times, goals):
        self.ts = self.ts[:-1] + times
        self.path = self.path[:-1] + goals

    def update_states(self, t):
        idx = bisect.bisect_right(self.ts, t) - 1
        t0 = self.ts[idx]
        self.t_curr = t
        self.goal_curr = self.path[idx+1]
        if self.path[idx] != self.path[idx+1]:    
            dir = self.path[idx+1] - self.path[idx]
            dir = dir / np.linalg.norm(dir)
            self.x_curr = self.path[idx] + dir * vmax * (t-t0)
        else:
            self.x_curr = self.path[idx]
        if self.chop_state[idx]:
            self.in_CHOP = True



if __name__ == '__main__':
    # np.random.seed(0)
    # starts = np.random.uniform(0,10,(8,2))
    # goals = np.random.uniform(0,10,(8,2))


    vmax = 1
    radius = 0.5
    starts = np.array([[-5,0],[0,-5],[-5, -5], [6, -3], [11, 2]])
    goals = np.array([[5,0],[0,5],[10, 10], [14, -3], [11, -7]])
    robots = []
    for i in range(len(starts)):
        robots.append(new_robot(i, starts[i,:], goals[i,:], vmax, radius))


    plan = planner(robots, radius, vmax)
    plan.check_collision()

    # # start_times, robots, starts, goals, radius, vmax
    chop = CHOP(4, [robots[0],robots[1],robots[2]], starts[[0,1,2],:], goals[[0,1,2],:], 0.5, 1)
    # chop = CHOP(4, [robots[0],robots[1]], starts[[0,1],:], goals[[0,1],:], 0.5, 1)
    chop.build_CHOP()

    plt.figure()
    plt.plot([starts[0,0], goals[0,0]], [starts[0,1], goals[0,1]])
    plt.plot([starts[1,0], goals[1,0]], [starts[1,1], goals[1,1]])
    plt.plot([starts[2,0], goals[2,0]], [starts[2,1], goals[2,1]])
    plt.plot([starts[3,0], goals[3,0]], [starts[3,1], goals[3,1]])
    plt.plot([starts[4,0], goals[4,0]], [starts[4,1], goals[4,1]])
    plt.plot(chop.Xm[:,0], chop.Xm[:,1])
    plt.legend(['0','1','2','3','4'])
    plt.show()


    while True:
        points = np.ones((self.nw, 2)) * path_r
        points[:,0] *= cs
        points[:,1] *= ss
        points += self.xc
        paths = np.zeros((self.nw, 2))
        dist_path = np.zeros(self.nw)
        for j in range(self.nw):
            if j:
                paths[j,:] = points[j,:] - points[j-1,:]
            else:
                paths[j,:] = points[j,:] - points[-1,:]
        dist_path = np.sqrt(path[:,0]**2 + path[:,0]**2)
        dist_node = np.zeros((self.nw, len(self.Xg)))
        dist_line = np.zeros((self.nw, len(self.Xg)))
        for i in range(len(self.Xg)):
            gx = self.Xg[i,0]
            gy = self.Xg[i,1]
            dist_node[:,i] = np.sqrt((points[:,0]-gx)**2 + (points[:,1]-gy)**2)
            for j in range(self.nw):
                d1 = dist_node[j,:]
                if j:
                    d2 = dist_node[j-1,:]
                else:
                    d2 = dist_node[-1,:]
                d3 = dist_path[j]
                if max((d1,d2,d3)) == d3:
                    d4 = (d1**2-d2**2+d3**2) / (2*d3)
                    dist_line[j,i] = np.sqrt(d1**2-d4**2)
                else:
                    dist_line[j,i] = min((d1,d2))

            if any(dist_node < min_dist_node):
                id_min = np.argmin(dist_node)
                path_r = np.sqrt((points[id_min,0]-self.xc[0])**2 + (points[id_min,1]-self.xc[1])**2) + min_dist_node
                continue
            dist_line = np.zeros(self.nw)
            for i in range(self.nw):
                if i:
                    d1 = ()





    






        # while True:
        #     for robot in self.robots:
        #         robot.update_states(self.t_curr)

        #     if self.CHOPs:
        #         t_min = 100000
        #         for chop in self.CHOPs:
        #             if chop.ts <= self.t_curr and chop.t_crit[-1] >= self.t_curr:
        #                 idx = bisect.bisect_left(chop.t_crit, self.t_curr)
        #                 t_temp = chop.t_crit[idx]
        #                 if t_temp < t_min:
        #                     t_min = t_temp
        #             # t_next = t_min
        #     else:
        #         dt_min = 0
        #         for idx in self.ids_OMP:
        #             dt_temp = np.linalg.norm(self.robots[idx].goal_curr - self.robots[idx].x_curr) / self.vmax
        #             if dt_temp < dt_min:
        #                 dt_min = dt_temp
        #     t_next = min(t_min, self.t_curr + dt_min)
     

        #     # search_radius = self.vmax * 2 * t_next
        #     ids_intersect = None
        #     ts_intersect = None
        #     for i in range(n):
        #         if not self.robots[i].in_CHOP:
        #             xi = self.robots[i].x_curr
        #             gi = self.robots[i].goal_curr
        #             vect_i = gi - xi
        #             nvi = vect_i / np.linalg.norm(vect_i) 
        #             for j in range(i+1, len(self.robots)):
        #                 if not self.robots[j].in_CHOP:
        #                     xj = self.robots[j].x_curr
        #                     gj = self.robots[j].goal_curr
        #                     vect_j = gj - xj
        #                     nvj = vect_j / np.linalg.norm(vect_j)
        #                     dnv = nvi - nvj
        #                     dx = xi - xj
        #                     a = dnv @ dnv
        #                     b = 2 * dnv @ dx
        #                     c = dx @ dx - 8 * self.robots[i].R**2
        #                     val = b**2 - 4*a*c
        #                     if val > 0:
        #                         t1 = (-b + np.sqrt(val)) / 2 / a
        #                         t2 = (-b - np.sqrt(val)) / 2 / a
        #                         self.collision_times[0,i,j] = min(t1, t2)
        #                         self.collision_times[1,i,j] = max(t1, t2)
        #                     else:
        #                         if c > 0:
        #                             self.collision_times[0,i,j] = -1
        #                             self.collision_times[1,i,j] = -1
        #                         else:
        #                             print('Robot {} and {} always intersect'.format(i,j))
        #                             return
        #                 else:
        #                     self.collision_times[0,i,j] = -1
        #                     self.collision_times[1,i,j] = -1

        #         else:
        #             self.collision_times[0,i,j] = -1
        #             self.collision_times[1,i,j] = -1
        #     mask = ((self.t_curr <= self.collision_times) * (self.collision_times <= t_next)).astype(bool)
        #     if mask.any():
        #         ts_intersect = min(self.collision_times[mask])
        #         masked_array = self.collision_times + ~mask * t_next
        #         ids_intersect = np.unravel_index(masked_array.argmin(), masked_array.shape)
        #         ids_intersect = ids_intersect[1:]

        #     if ts_intersect is not None:
        #         robots_in = [self.robots[x] for x in ids_intersect]
        #         self.ids_OMP = {idx for idx in self.ids_OMP if idx not in robots_in}
        #         self.ids_CHOP = {idx for idx in robots_in}
        #         n = len(robots_in)
        #         starts = np.zeros((n,2))
        #         goals = np.zeros((n,2))
        #         for i in range(n):
        #             idx = ids_intersect[i]
        #             dir = self.robots[idx].goal_curr - self.robots[idx].x_curr
        #             starts[i,:] = self.robots[idx].x_curr + dir / np.linalg.norm(dir) * (ts_intersect - self.t_curr) * self.vmax
        #             goals[i,:] = self.robots[idx].goal
        #         new_chop = CHOP(ts_intersect, robots_in, starts, goals, self.R, self.vmax)
        #         new_chop.build_CHOP()
        #         # self.CHOPs.append(new_chop)

        #         while True:
        #             idx_add = {}
        #             t_max = new_chop.t_chop[-1]
        #             for idx in self.ids_OMP:
        #                 x1 = self.robots[idx].x_curr
        #                 dt = t_max - self.robots[idx].t_curr
        #                 dir = (self.robots[idx].x_curr - self.robots[idx].goal_curr)
        #                 dx = dir/np.linalg.norm(dir) * dt
        #                 x2 = x1 + dx
        #                 d3 = dx @ dx
        #                 d1 = np.linalg.norm(x1 - new_chop.xc)
        #                 d2 = np.linalg.norm(x2 - new_chop.xc)
        #                 alpha = (d1**2-d2**2+d3**2) / (2*d3)
        #                 if (d3 - alpha) > d3 or alpha > d3:
        #                     dist = min(d1,d2)
        #                 else:
        #                     dist = np.sqrt(d1**2 - alpha**2)
        #                 if dist < 2*self.R:
        #                     idx_add.add(idx)
                    
        #             chop_add = {}
        #             for chop in self.CHOPs:
        #                 if (chop.ts >= new_chop.ts and chop.ts <= new_chop.t_chop) or (chop.t_chop >= new_chop.ts and chop.t_chop <= new_chop.t_chop):
        #                     dist = np.linalg.norm(chop.xc - new_chop.xc) - chop.r - new_chop.r
        #                     shared_ids = [robot.id for robot in chop.robots if robot in new_chop.robots]
        #                     if dist < 2*self.R or shared_ids:
        #                         chop_add.add(chop)
        #                         idx_chop = {robot.idx for robot in chop.robots}
        #                         idx_add.add(idx_chop)
        #                     for i in len(chop.t_crit[:-1]):
        #                         t = chop.t_crit[i]
        #                         if (t >= new_chop.ts and t <= new_chop.t_chop):
        #                             x1 = chop.goals[i][-2]
        #                             t_max = min(new_chop.t_chop, chop.t_list[i][-1])
        #                             dt = t_max - t
        #                             dir = (chop.goals[i][-1] - chop.goals[i][-2])
        #                             dx = dir/np.linalg.norm(dir) * dt
        #                             x2 = x1 + dx
        #                             d3 = dx @ dx
        #                             d1 = np.linalg.norm(x1 - new_chop.xc)
        #                             d2 = np.linalg.norm(x2 - new_chop.xc)
        #                             alpha = (d1**2-d2**2+d3**2) / (2*d3)
        #                             if (d3 - alpha) > d3 or alpha > d3:
        #                                 dist = min(d1,d2)
        #                             else:
        #                                 dist = np.sqrt(d1**2 - alpha**2)
        #                             if dist < 2*self.R:
        #                                 idx = chop.robots[i].id
        #                                 idx_add.add(idx)
                            
        #                 shared_ids = [robot.id for robot in chop.robots if robot in new_chop.robots]
        #                 if len(shared_ids) >= 2:
        #                     chop_add.add(chop)
        #                     idx_chop = {robot.idx for robot in chop.robots}
        #                     idx_add.add(idx_chop)
        #             if chop_add:
        #                 t_min = min([chop.ts for chop in chop_add] + [new_chop])
        #                 self.update_states(t_min)
        #             if idx_add:
        #                 robots_in.add({self.robots[idx] for idx in idx_add})
        #                 n = len(robots_in)
        #                 starts = np.zeros((n,2))
        #                 goals = np.zeros((n,2))
        #                 for i in range(n):
        #                     starts[i,:] = self.robots[i].x_curr
        #                     goals[i,:] = self.robots[i].goal_curr
        #                 new_chop = CHOP(t_min, robots_in, starts, goals, self.R. self.vmax)
        #                 # self.CHOPs.append(new_chop)
        #             else:

        #                 break
