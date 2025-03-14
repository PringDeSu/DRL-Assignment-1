import numpy as np
"""
obs:
    (0, 1)              => agent_pos
    (2, 3)              => coor1
    (4, 5)              => coor2
    (6, 7)              => coor3
    (8, 9)              => coor4
    (10, 11, 12, 13)    => obstacle
    14                  => has_passenger
    15                  => has_destination
"""
class ExternalAPI:
    def __init__(self):
        self.current_stage = None
        self.ppp = []
        self.pdp = []
        self.pre_obs = None
        self.pre_action = None
        self.pre_coor = ((-1, -1), (-1, -1), (-1, -1), (-1, -1))

    def remove_impossible(self, obs):
        if obs[14] == 0:
            sz = len(self.ppp)
            for i in reversed(range(sz)):
                if abs(self.ppp[i][0] - obs[0]) + abs(self.ppp[i][1] - obs[1]) <= 1:
                    self.ppp.pop(i)
        if obs[15] == 0:
            sz = len(self.pdp)
            for i in reversed(range(sz)):
                if abs(self.pdp[i][0] - obs[0]) + abs(self.pdp[i][1] - obs[1]) <= 1:
                    self.pdp.pop(i)

    def min_dis_rel_pos(self, ls, tg):
        dis = np.zeros(len(ls))
        for i in range(len(ls)):
            dis[i] = abs(ls[i][0] - tg[0]) + abs(ls[i][1] - tg[1])
        big_id = dis.argmin()
        return (ls[big_id][0] - tg[0], ls[big_id][1] - tg[1])

    def reset(self, obs):
        self.current_stage = 0
        self.pre_obs = obs
        self.ppp = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        self.pdp = [(obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])]
        self.pre_coor = ((obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9]))
        self.remove_impossible(obs)
        return (self.current_stage, self.min_dis_rel_pos(self.ppp, (obs[0], obs[1])), (obs[10], obs[11], obs[12], obs[13]))

    def add_action(self, action):
        self.pre_action = action

    def GOTO_STAGE0(self):
        self.current_stage = 0
        self.ppp = [(self.pre_obs[0], self.pre_obs[1])]

    def GOTO_STAGE1(self):
        self.current_stage = 1
        self.ppp = []

    def get_state(self, obs):
        if ((obs[2], obs[3]), (obs[4], obs[5]), (obs[6], obs[7]), (obs[8], obs[9])) != self.pre_coor:
            return self.reset(obs)

        self.remove_impossible(obs)
        if self.current_stage == 0:
            if self.pre_action == 4:
                if (self.pre_obs[0], self.pre_obs[1]) in self.ppp:
                    self.GOTO_STAGE1()
        else:
            if self.pre_action == 5:
                self.GOTO_STAGE0()
        state = [self.current_stage, None, (obs[10], obs[11], obs[12], obs[13])]
        if self.current_stage == 0:
            state[1] = self.min_dis_rel_pos(self.ppp, (obs[0], obs[1]))
        else:
            state[1] = self.min_dis_rel_pos(self.pdp, (obs[0], obs[1]))
        self.pre_obs = obs
        self.pre_action = None
        return tuple(state)

