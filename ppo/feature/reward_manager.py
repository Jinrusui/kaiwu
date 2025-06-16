#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
import math
import numpy as np
from ppo.config import GameConfig
from collections import deque

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.m_last_frame_totalHurtToHero =0#add
        self.m_last_frame_totalBeHurtByHero =0#add
        self.m_last_frame_soldier_av_hp = -1#add
        self.main_soldiers = []
        self.enemy_soldiers = []
        self.m_last_frame_hp = -1 # added
        self.m_last_frame_grass_status = False # added
        self.m_last_frame_received_enemy_hurt = -1 # added
        self.grass_position_list = [] # added
        self.m_last_frame_pos = [] #add
        self.m_last_frame_target = None
        self.last_few_frame_hp = deque(maxlen=8)
        self.combat_status=0
        self.combat_rec = deque(maxlen=10)
        self.cake_avaliable=0
        self.last_frame_dist_hero2eh = 0
        self.aggressive = 0
        self.org_atk_spd = 0
        self.org_crit_rate = 0
        self.game_status = "default"
        # self.game_status ={}
        # self.game_status_rec_dict={}
        self.game_status_dict = {
            "defend": -2,
            "runaway": -1,
            "attackTower": 2,
            "combat": 0.5,
            "default": 0
        }

        self.status_mask = {
            "defend": {
                "hp_point": 1,
                "tower_hp_point": 1,
                "attack_tower": 0,
                "defend_tower": 1,
                "approach_tower": 1,
                "money": 0,
                "exp": 0,
                "ep_rate": 0,
                "death": 1,
                "kill": 1,
                "last_hit": 1,
                "forward": 0,
                "HurtToHero": 1,
                "BeHurtByHero": 1,
                'retreat':0,
                'heal': 1,
                'skill_hit_count': 1,
                'useSkillToHero': 1,
                'flash': 0,
                'strengthen':1,
                'enemy_sodiler_hp':1,
                'enemy_sodiler_pos':1,
                "HurtToOthers":0,
                "game_status":1,
                "skill_hit_count_bonus":1
                
            },
            "runaway": {
                "hp_point": 1,
                "tower_hp_point": 0,
                "attack_tower": 0,
                "defend_tower": 0,
                "approach_tower": 0,
                "money": 0,
                "exp": 0,
                "ep_rate": 0,
                "death": 1,
                "kill": 0,
                "last_hit": 0,
                "forward": 0,
                "HurtToHero": 0,
                "BeHurtByHero": 0,
                'retreat': 1,
                'heal': 1,
                'skill_hit_count': 0,
                'useSkillToHero': 0,
                'flash': 1,
                'strengthen': 0,
                'enemy_sodiler_hp':0,
                'enemy_sodiler_pos':0,
                "HurtToOthers":0,
                "game_status":1,
                "skill_hit_count_bonus":0
            },
            "attackTower": {
                "hp_point": 1,
                "tower_hp_point": 1,
                "attack_tower": 1,
                "defend_tower": 0,
                "approach_tower": 1,
                "money": 0,
                "exp": 0,
                "ep_rate": 0,
                "death": 1,
                "kill": 1,
                "last_hit": 0,
                "forward": 0,
                "HurtToHero": 0,
                "BeHurtByHero": 1,
                'retreat': 0,
                'heal': 0,
                'skill_hit_count': 0,
                'useSkillToHero': 0,
                'flash': 0,
                'strengthen': 1,
                'enemy_sodiler_hp':0,
                'enemy_sodiler_pos':0,
                "HurtToOthers":0,
                "game_status":1,
                "skill_hit_count_bonus":0
            },
            "combat": {
                "hp_point": 1,
                "tower_hp_point": 0,
                "attack_tower": 0,
                "defend_tower": 0,
                "approach_tower": 0,
                "money": 1,
                "exp": 1,
                "ep_rate": 1,
                "death": 1,
                "kill": 1,
                "last_hit": 0,
                "forward": 1,
                "HurtToHero": 1,
                "BeHurtByHero": 1,
                'retreat': 0,
                'heal': 1,
                'skill_hit_count': 1,
                'useSkillToHero': 1,
                'flash': 1,
                'strengthen':1,
                'enemy_sodiler_hp':0,
                'enemy_sodiler_pos':0,
                "HurtToOthers":0,
                "game_status":1,
                "skill_hit_count_bonus":1

                
                
            },
            "default": {
                "hp_point": 1,
                "tower_hp_point": 0,
                "attack_tower": 0,
                "defend_tower": 1,
                "approach_tower": 0,
                "money": 1,
                "exp": 1,
                "ep_rate": 1,
                "death": 1,
                "kill": 0,
                "last_hit": 1,
                "forward": 1,
                "HurtToHero": 0,
                "BeHurtByHero": 0,
                'retreat': 1,
                'heal': 1,
                'skill_hit_count': 1,
                'useSkillToHero': 0,
                'flash': 0,
                'strengthen':1,
                'enemy_sodiler_hp':1,
                'enemy_sodiler_pos':1,
                "HurtToOthers":1,
                "game_status":1,
                "skill_hit_count_bonus":1
                
            }
        }
        

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        # self.game_status_rec()
        self.get_reward(frame_data, self.m_reward_value)
        self.last_frame_data_process(frame_data)

        frame_no = frame_data["frameNo"]
        self.m_last_frame_no = frame_no-1
        if self.time_scale_arg > 0:
            for key in self.m_reward_value:
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    # main_hero, enemy_hero = None, None
    # main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        #global main_hero, enemy_hero, main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers, enemy_soldiers
        
        # Get both agents
        # 获取双方智能体
        #print(self.last_few_frame_hp)
        
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
                
        #print(main_hero['actor_state']['runtime_id'],self.main_hero_player_id)
        main_hero_hp = main_hero["actor_state"]["hp"]
        #print(main_hero_hp)
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_hp_rate = main_hero_hp/main_hero_max_hp
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        enemy_hero_hp = enemy_hero["actor_state"]["hp"]
        #print(main_hero_hp)
        enemy_hero_max_hp = enemy_hero["actor_state"]["max_hp"]
        enemy_hero_hp_rate = enemy_hero_hp/enemy_hero_max_hp
        enemy_hero_runtime_id = enemy_hero['actor_state']['runtime_id']
        #print('enemyid',enemy_hero_runtime_id)
        runtime_id = main_hero['actor_state']['runtime_id']
        
        
        
        if frame_data.get('cakes',None):
            self.cake_avaliable = 1 if frame_data['cakes'][0]['collider']['location']['x']>0 else 0
        
        # Get both defense towers
        # 获取双方防御塔和小兵
        main_tower, main_spring, enemy_tower, enemy_spring, main_soldiers,enemy_soldiers = None, None, None, None,[],[]
        self.main_soldiers.clear()
        self.enemy_soldiers.clear()

        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.main_soldiers.append(organ)
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ
                elif organ_subtype == "ACTOR_SUB_SOLDIER":
                    self.enemy_soldiers.append(organ)
        
        #combat status
        hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
        eh_hit_target_info = enemy_hero["actor_state"].get("hit_target_info", None)
        if hit_target_info:
            hit_list = [hit_tar['hit_target'] for hit_tar in hit_target_info ]
        if eh_hit_target_info:
            eh_hit_list = [hit_tar['hit_target'] for hit_tar in eh_hit_target_info ]
    
        if (main_hero["totalHurtToHero"] - self.m_last_frame_totalHurtToHero)>0 or (main_hero["totalBeHurtByHero"] - self.m_last_frame_totalBeHurtByHero)>0:
            self.combat_rec.append(1)
        else:
            self.combat_rec.append(0)
        self.combat_status=1 if (1 in self.combat_rec and enemy_hero_hp_rate>0 and main_hero_hp_rate>0) else 0
        #print(f'combat:{self.combat_status}')
        #print(f'hit_tar_info:{hit_target_info}\n')
        
        

        #     print(enemy_hero_runtime_id)
        # if enemy_hero_runtime_id ==main_hero['actor_state'].get('attack_target',None):
        #     print('attack_tag:',main_hero['actor_state']['attack_target'])
        # print('behurtseq:',main_hero.get('takeHurtInfos',None))
        tower_hit_tar_info =  main_tower['attack_target']
        dist_hero2eh = self.calculate_distance(main_hero["actor_state"]["location"],enemy_hero["actor_state"]["location"])

        idle_skill = sum(main_hero['skill_state']['slot_states'][n]['usable'] for n in [1,2,3]) 
        
        #print('use:',use_skill,'usable:',idle_skill)
        enemy_hit_tar_info =  enemy_tower['attack_target']
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        #main_spring_pos = (main_spring["location"]["x"], main_spring["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        enemy_hero_pos = (
            enemy_hero["actor_state"]["location"]["x"],
            enemy_hero["actor_state"]["location"]["z"],
        )
        #dist_hero2spring = math.dist(hero_pos, main_spring_pos)
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        dist_eh2main = math.dist(enemy_hero_pos, main_tower_pos)
        #dist_main2spring = math.dist(main_spring_pos,main_tower_pos)
        #dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        
         
        #compute aggressive
        reward =0
        # Consider level advantage
        level_diff = main_hero['level'] - enemy_hero['level']
        #if level_diff > 0:
        reward +=  0.15 * level_diff  # Reward higher-level hero attacking lower-level enemy
        
        # Consider money advantage
        if enemy_hero['money']!=0 and main_hero['money']!=0:
                    money_rate = main_hero['money']/enemy_hero['money']
        else:money_rate = 0

        if money_rate > 1:
            reward +=  money_rate**1.5 -1
        else:
            reward -= 1-math.sqrt(money_rate)#**1.5 

        # Compare HP percentages######
        main_hero_hp_percentage = main_hero_hp_rate#main_hero_hp / max(main_hero_max_hp, 1)**2
        enemy_hero_hp_percentage = enemy_hero_hp_rate#enemy_hero["actor_state"]["hp"] / max(enemy_hero["actor_state"]["max_hp"], 1)**2
        if main_hero_hp_percentage!=0 and enemy_hero_hp_percentage!=0:
                    hp_rate = main_hero_hp_percentage/enemy_hero_hp_percentage
        else:hp_rate = 0
        
        if hp_rate > 1:
            reward +=  (hp_rate-1)*1.2*1.5 # Reward higher-level hero attacking lower-level enemy
        else:
            reward -=  (1-hp_rate**1.5)*1.5

        # Bonus if in grass
        if main_hero.get("isInGrass", False):
            reward += 0.05
              # Encourage using advantageous positions
        
        #是否处于强化形态
        if main_hero['actor_state']['config_id']==508:
            
            if 9000<main_hero['actor_state']['attack_range']:#enemy_hero['actor_state']['attack_range']<dist_hero2eh<=main_hero['actor_state']['attack_range'] :
                reward+=0.08
            if main_hero['actor_state']['values']['crit_rate']>3000:
                reward+=0.08
        #print(main_hero['actor_state']['attack_range'])
        #if use_skill:
            #print(f'org_atk_rag:{self.org_atk_spd},atk_spd:',main_hero['actor_state']['values']['atk_spd'])
            #print(main_hero['actor_state']['buff_state'])
        if main_hero['actor_state']['config_id']==199 and 6900>main_hero['actor_state']['attack_range']:
            reward+=0.15
    
        # Bonus if allied soldiers are attacking the enemy
        if self.main_soldiers:
            soldiers_hitting_enemy = 0
            #hit_targets = [hit_tar['hit_target'] for hit_tar in hit_target_info]:
            for soldier in self.main_soldiers:
                s_hit_target_info = soldier.get("hit_target_info", None)
                if s_hit_target_info and enemy_hero_runtime_id in s_hit_target_info :
                    soldiers_hitting_enemy += 1
            reward +=  0.04 * soldiers_hitting_enemy
        
        if self.enemy_soldiers:
            soldiers_hitting_main = 0
            #hit_targets = [hit_tar['hit_target'] for hit_tar in hit_target_info]
            for soldier in self.enemy_soldiers:
                s_hit_target_info = soldier.get("hit_target_info", None)
                if s_hit_target_info and runtime_id in s_hit_target_info :
                    soldiers_hitting_main += 1
            reward -=  0.04 * soldiers_hitting_main

        if dist_hero2emy <= 8800:
            reward-=0.5
        if dist_eh2main <= 8800:
            reward+=0.5
        
        if self.combat_status ==0:
            if idle_skill:
                reward +=0.05*idle_skill
            else:
                reward-=0.2

        self.aggressive = reward
        #combat status
        def combat_status():
            if self.combat_status==1:
                return True
            else:
                return False

        def attack_tower_status(enemy_tower,main_hero_hp_rate,enemy_hero_hp_rate):
            enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
            
            #enemy_hit_tar_info = enemy_tower['attack_target']
            if_enemy_hit_soldier = False
            if self.main_soldiers != []:
                
                for soldier in self.main_soldiers:
                    s_pos = (soldier["location"]["x"],soldier["location"]["z"])
                    dist_s2emy = math.dist(s_pos,enemy_tower_pos)
                    if  dist_s2emy<=9000:
                        if_enemy_hit_soldier = True
                    #     if frame_data.get('bullets',None) :
                    #         for bullet in frame_data['bullets']:
                    #             if bullet['source_actor'] == enemy_tower['runtime_id'] and bullet.get('target_actor',None) and bullet['target_actor'] == soldier['runtime_id']:
                    #                 print(1)
                        # if_hit_tower +=1
                        # if soldier['runtime_id'] == enemy_hit_tar_info:
                    
            if (main_hero_hp_rate>0.6 or enemy_hero_hp_rate<0.4)and if_enemy_hit_soldier:
                return True
            return False

        def defend_status(main_tower):
            #main_hit_tar_info = main_tower['attack_target']
            main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
            if self.enemy_soldiers != []:
                for soldier in self.enemy_soldiers:
                    es_pos = (soldier["location"]["x"],soldier["location"]["z"])
                    dist_es2main = math.dist(es_pos,main_tower_pos)
                    # for soldier in self.enemy_soldiers:
                    #     if frame_data.get('bullets',None) :
                    #         for bullet in frame_data['bullets']:
                    #             if bullet['source_actor'] == main_tower['runtime_id'] and bullet.get('target_actor',None) and bullet['target_actor'] == soldier['runtime_id']:
                    #                 print(2)
                    if dist_es2main<=8800:
                        return True
                        #if soldier['runtime_id'] == main_hit_tar_info:
                            
            return False
        
        def runaway_status(main_hero_hp_rate,enemy_hero_hp_rate):
            if self.combat_status==1 and self.aggressive<0:
                return True
            elif self.combat_status==0 and main_hero_hp_percentage<0.3:
                return True
            else: return False


        def change_game_status():
            if defend_status(main_tower):
                
                game_status = "defend"
                
            elif runaway_status(main_hero_hp_rate,enemy_hero_hp_rate):
                game_status = "runaway"

            elif attack_tower_status(enemy_tower,main_hero_hp_rate,enemy_hero_hp_rate):
                game_status = "attackTower"
            
            elif combat_status():
                game_status = "combat"
            
            else:
                game_status = "default"
            self.game_status = game_status

        change_game_status()
        
        ######record hurt####
        # print('h2h:',main_hero["totalHurtToHero"])
        # print('l_h2h:',self.m_last_frame_totalHurtToHero)
        # print('h2h:',main_hero["totalBeHurtByHero"])
        # print('l_h2h:',self.m_last_frame_totalBeHurtByHero)
        # print(self.hurt2hero_rec,self.hurtbyhero_rec)
        # self.hurt2hero_rec.append((main_hero["totalHurtToHero"] - self.m_last_frame_totalHurtToHero)/max(1,enemy_hero_max_hp))
        # self.hurtbyhero_rec.append((main_hero["totalBeHurtByHero"] - self.m_last_frame_totalBeHurtByHero)/max(1,main_hero_max_hp))
        # self.hurt_eff=sum(np.array(self.hurt2hero_rec)-np.array(self.hurtbyhero_rec))
        #print('hut_eff:',self.hurt_eff)



        #     print('hit_list:',hit_list,'\n')
        # print(f'hero_skill:',main_hero['skill_state']['slot_states'])
        # print('hit_target_info:',hit_target_info,'\n')
        #print(main_hero['actor_state']['attack_range'])

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            
            # Money
            # 金钱
            
            if reward_name == "money":
                
                reward_struct.cur_frame_value = main_hero["moneyCnt"]
                
            
            elif reward_name == "attack_tower":
                reward_struct.cur_frame_value =1.0 * enemy_tower['hp']/enemy_tower['max_hp']
                
            elif reward_name == "approach_tower":
                hero_pos = (
                    main_hero["actor_state"]["location"]["x"],
                    main_hero["actor_state"]["location"]["z"],
                )
                main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
                dist_hero2main = math.dist(hero_pos, main_tower_pos)
                enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
                dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
                if self.game_status == "defend" and dist_hero2main>8800:
                    reward_struct.cur_frame_value = dist_hero2main

                elif self.game_status == "attacktower" and dist_hero2emy >main_hero['actor_state']['attack_range']:
                    reward_struct.cur_frame_value = dist_hero2emy*10
                else:reward_struct.cur_frame_value = 0

            elif reward_name == "defend_tower":
                main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
                enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
                main_spring_pos = (main_spring["location"]["x"], main_spring["location"]["z"])
                hero_pos = (
                    main_hero["actor_state"]["location"]["x"],
                    main_hero["actor_state"]["location"]["z"],
                )
                dist_hero2spring = math.dist(hero_pos, main_spring_pos)
                dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
                dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
                dist_main2spring = math.dist(main_spring_pos,main_tower_pos)
                main_hit_tar_info = main_tower['attack_target']
                hit_target_info = main_hero["actor_state"].get("hit_target_info", None)
                if_main_hit_soldier = 0
                if_hit_soldier = 0
                defend_value = 0
                if self.enemy_soldiers != []:
                    for soldier in self.enemy_soldiers:
                            # if frame_data.get('bullets',None) :
                            #     for bullet in frame_data['bullets']:
                            #         if bullet['source_actor'] == main_hero_runtime_id and bullet.get('target_actor',None) and bullet['target_actor'] == soldier['runtime_id']:
                            #             if_hit_soldier +=1
                                        
                            #         if bullet['source_actor'] == enemy_tower['runtime_id'] and bullet.get('target_actor',None) and bullet['target_actor'] == soldier['runtime_id']:
                            #             if_main_hit_soldier +=1
                            # if soldier['runtime_id'] ==main_hero['actor_state']['attack_target']:
                            #     if_hit_soldier +=1
                        if hit_target_info:
                            hit_list = [hit_tar['hit_target'] for hit_tar in hit_target_info ]
                            if soldier['runtime_id'] in hit_list:
                                if_hit_soldier +=1

                if if_hit_soldier == 0:
                    defend_value -= self.m_last_frame_soldier_av_hp*len(self.enemy_soldiers)
                reward_struct.cur_frame_value = defend_value

            elif reward_name == "game_status":
                reward_struct.cur_frame_value = self.game_status_dict[self.game_status]

            elif reward_name == "retreat":
                # Go back to spring to recover
                main_spring_pos = (main_spring["location"]["x"], main_spring["location"]["z"])
                hero_pos = (
                    main_hero["actor_state"]["location"]["x"],
                    main_hero["actor_state"]["location"]["z"],
                )
                dist_hero2spring = math.dist(hero_pos, main_spring_pos)
                dist_hero2cake = math.dist(hero_pos,(15340,15100))
                if main_hero["actor_state"]["hp"] / max(main_hero["actor_state"]["max_hp"],1) <= 0.35:
                    reward_struct.cur_frame_value = dist_hero2spring
                elif main_hero["actor_state"]["hp"] / max(main_hero["actor_state"]["max_hp"],1)<0.5:
                    if self.cake_avaliable:
                        reward_struct.cur_frame_value = dist_hero2cake
                    else:
                        reward_struct.cur_frame_value = dist_hero2spring
                else:reward_struct.cur_frame_value = 0
            # Health points
            # 生命值
                """
                还是存在被后面复杂reward影响的情况，并不是占主导的reward，在效果上agent对于血量控制并不好，
                但是总体比money要存在感强一些。
                """
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
                """
                不是重点reward

                """
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
                """
                与hp相似

                """
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
                """
                与hp相似
                """
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
                """
                理论上最重要的reward，但是学的不好
                """
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
                # if self.game_status == "attackTower":
                #     print(main_hero_runtime_id,reward_struct.cur_frame_value)
                
            # Last hit
            # 补刀
                """
                补刀还不错

                """
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
                
            # Experience points
            # 经验值
                """
                与 money类似
                """
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower,main_spring,enemy_hero)
                """
                学会了不乱用治疗
                """
            elif reward_name == "heal":
                reward_struct.cur_frame_value = main_hero['skill_state']['slot_states'][4]['usedTimes']
                """
                没学会，基本不用技能（可能把普攻算技能了？）
                """
            elif reward_name == "skill_hit_count":
                if main_hero['actor_state']['config_id']==133:
                    reward_struct.cur_frame_value = sum(main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] for n in [1,2,3])
                elif main_hero['actor_state']['config_id']==199:
                    if 7000>main_hero['actor_state']['attack_range']:
                        reward_struct.cur_frame_value = sum((main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] / 8 if n == 0 else main_hero['skill_state']['slot_states'][n]['hitHeroTimes']) for n in [0, 1, 2, 3])
                    else:
                        reward_struct.cur_frame_value = sum(main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] for n in [1,2,3])

                elif main_hero['actor_state']['config_id']==508:
                    if 9000<main_hero['actor_state']['attack_range'] or main_hero['actor_state']['values']['crit_rate']>3000:
                        reward_struct.cur_frame_value = sum((main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] / 8 if n == 0 else main_hero['skill_state']['slot_states'][n]['hitHeroTimes']) for n in [0, 1, 2, 3])
                    else:
                        reward_struct.cur_frame_value = sum(main_hero['skill_state']['slot_states'][n]['hitHeroTimes'] for n in [1,2,3])
            
            elif reward_name == "skill_hit_count_bonus":
                if main_hero['actor_state']['config_id']==133:
                    reward_struct.cur_frame_value = main_hero['skill_state']['slot_states'][3]['hitHeroTimes']
                elif main_hero['actor_state']['config_id']==199:
                    reward_struct.cur_frame_value = main_hero['skill_state']['slot_states'][3]['hitHeroTimes']
                else:
                    if 9000<main_hero['actor_state']['attack_range'] or main_hero['actor_state']['values']['crit_rate']>3000:
                        reward_struct.cur_frame_value = main_hero['skill_state']['slot_states'][0]['hitHeroTimes']/5
                    else:0

            #对英雄输出
               
                       
            elif reward_name == "HurtToHero":
                
                reward_struct.cur_frame_value = main_hero["totalHurtToHero"]/enemy_hero_max_hp


               
            elif reward_name == "HurtToOthers":
                # Calculate the base reward: damage to non-hero units, normalized
                damage_to_others = main_hero["totalHurt"] - main_hero["totalHurtToHero"]
                base_reward = damage_to_others / 10000  # Normalization factor

                # Initialize balance_rate
                balance_rate = 1.0

                # Increase balance_rate based on continuous hit count to encourage combos
                if hit_target_info:
                    conti_hit_counts = [
                        item.get('conti_hit_count', 0) for item in hit_target_info
                    ]
                    max_conti_hit_count = max(conti_hit_counts, default=0)
                    balance_rate += 0.03 * max_conti_hit_count  # Modest increase per hit

                # Adjust reward based on hero's current HP percentage
                hp_percentage = main_hero_hp / max(main_hero_max_hp, 1)
                reward = base_reward * balance_rate * hp_percentage

                reward_struct.cur_frame_value = reward


        ####    # #承受英雄伤害
                """ 体现不太出来，不是很好学"""
            elif reward_name == "BeHurtByHero":
                # Initialize balance_rate
                reward_struct.cur_frame_value =main_hero["totalBeHurtByHero"]/main_hero_max_hp



            elif reward_name == "useSkillToHero":
                if self.combat_status:
                    if main_hero['actor_state']['config_id']==133:
                        reward_struct.cur_frame_value=idle_skill
                
                else:reward_struct.cur_frame_value=0

            
              
                                   
                """
                存在bug的reward，不过被forward的那个奖励代替了？
                """
    
            elif reward_name =='flash':
                if_use_flash=main_hero['skill_state']['slot_states'][-2]['succUsedInFrame']
                #print(if_use_flash)
                if_away = if_use_flash and self.last_frame_dist_hero2eh < enemy_hero['actor_state']['attack_range'] and dist_hero2eh>enemy_hero['actor_state']['attack_range'] and main_hero_hp_rate<0.1 and enemy_hero_hp_rate>0.5 and self.aggressive<0
                if_chase = if_use_flash and self.last_frame_dist_hero2eh > main_hero['actor_state']['attack_range'] and dist_hero2eh > main_hero['actor_state']['attack_range'] and main_hero_hp_rate>0.5 and enemy_hero_hp_rate<0.2 and self.aggressive>0
                if if_use_flash and (if_away or if_chase):
                    reward_struct.cur_frame_value = 1
                elif if_use_flash and not (if_away or if_chase):
                    reward_struct.cur_frame_value = -1
                else:
                    reward_struct.cur_frame_value = 0
            
           
            
            elif reward_name =='strengthen':
                if self.combat_status:
                    if main_hero['actor_state']['config_id'] ==508:
                        if 9000<main_hero['actor_state']['attack_range'] or main_hero['actor_state']['values']['crit_rate']>3000:
                            reward_struct.cur_frame_value=(9000<main_hero['actor_state']['attack_range'])*0.5+(main_hero['actor_state']['values']['crit_rate']>3000)*0.5
                        else:reward_struct.cur_frame_value=-0.5
                       
                    elif main_hero['actor_state']['config_id'] ==199:
                        if 6900>main_hero['actor_state']['attack_range'] :
                            reward_struct.cur_frame_value=0.3
                        else:reward_struct.cur_frame_value=-0.3
                else:
                    reward_struct.cur_frame_value=0
            
            elif reward_name =='enemy_sodiler_hp':
                if self.enemy_soldiers != []:
                    total_hp = 0
                    for soldier in self.enemy_soldiers:
                        total_hp+=soldier['hp']/soldier['max_hp']
                    reward_struct.cur_frame_value =total_hp
                else: reward_struct.cur_frame_value = 0
            
            elif reward_name =='enemy_sodiler_pos':
                reward_struct.cur_frame_value=0
                if self.enemy_soldiers!=[]:
                    es_pos = (self.enemy_soldiers[0]["location"]["x"],self.enemy_soldiers[0]["location"]["z"])
                    dist_es2main = math.dist(es_pos,main_tower_pos)
                    #鼓励提前清兵 
                    if dist_es2main<0.5*dist_main2emy:
                        reward_struct.cur_frame_value = -total_hp/len(self.enemy_soldiers)#math.sqrt(self.m_last_frame_soldier_av_hp*len(self.enemy_soldiers))
    
    
    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower,main_spring,enemy_hero):
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        main_spring_pos = (main_spring["location"]["x"], main_spring["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        forward_value = 0
        #dist_hero2spring = math.dist(hero_pos, main_spring_pos)
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        #dist_main2spring = math.dist(main_spring_pos,main_tower_pos)
        balance_rate = 1
        #战场前
        if dist_hero2emy > dist_main2emy:
            #进入战场
            if main_hero["actor_state"]["hp"] /max( main_hero["actor_state"]["max_hp"],1) > 0.99:
                forward_value += (dist_main2emy - dist_hero2emy) / dist_main2emy
            else:
                #惩罚在后方猫着
                if main_hero['actor_state']['hp'] and (main_hero["actor_state"]['hp']/main_hero["actor_state"]['max_hp'])>0.55:
                    forward_value += (dist_main2emy - dist_hero2emy) / dist_main2emy

        
        if enemy_tower['attack_target'] == main_hero['actor_state']['runtime_id'] :
            forward_value += (dist_hero2emy -8800)/ dist_main2emy*1000
          
        return forward_value
    
    
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        
    
###########################################################
    def last_frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        for hero in frame_data["hero_states"]:
            enemy = frame_data["hero_states"][1] if hero == frame_data["hero_states"][0] else frame_data["hero_states"][0]
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.m_last_frame_hp = hero["actor_state"]["hp"]
                if hero["actor_state"].get("hit_target_info", None) is not None:
                    self.m_last_frame_target = hero["actor_state"].get("hit_target_info", None)[0]['hit_target']  
                else:
                    self.m_last_frame_target = None
                self.m_last_frame_pos = (hero["actor_state"]["location"]["x"],hero["actor_state"]["location"]["z"])
                self.m_last_frame_grass_status = hero["isInGrass"]
                self.main_hero_camp = main_camp
                self.m_last_frame_totalHurtToHero =hero["totalHurtToHero"]
                self.m_last_frame_totalBeHurtByHero =hero["totalBeHurtByHero"]
                self.last_frame_dist_hero2eh = self.calculate_distance(hero["actor_state"]["location"],enemy["actor_state"]["location"])
                if hero["actor_state"]["hp"]== 0:
                    self.last_few_frame_hp = deque(maxlen=8)
                else:
                    self.last_few_frame_hp.append(hero["actor_state"]["hp"]/hero["actor_state"]["max_hp"])
                #print(self.last_few_frame_hp)
                if self.enemy_soldiers != []:
                    total_hp = 0
                    for soldier in self.enemy_soldiers:
                        total_hp+=soldier['hp']/soldier['max_hp']
                    self.m_last_frame_soldier_av_hp = total_hp/len(self.enemy_soldiers)
                else:self.m_last_frame_soldier_av_hp = 0
                #print(self.m_last_frame_soldier_av_hp,self.enemy_soldiers)
            else:
                enemy_camp = hero["actor_state"]["camp"]
#######################################################################

#################################################################
    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def calculate_distance(self,loc1, loc2):
        return math.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['z'] - loc2['z'])**2)

   
    
    # 判断是否连续多帧大幅度掉血
    def check_hp(self, hp_queue):
        if len(hp_queue) >1 and hp_queue[len(hp_queue)-1] <= 0.4 * hp_queue[0]:
            return hp_queue[len(hp_queue)-1]-hp_queue[0]
        elif len(hp_queue) >1 and hp_queue[len(hp_queue)-2] >= 1.2 * hp_queue[0]:
            return hp_queue[len(hp_queue)-2] - hp_queue[0]
        else:
            return 0
    

    
##################################################################  
    def get_reward(self, frame_data, reward_dict):
        #print(self.game_status)
        # status=self.game_status_rec_dict[self.main_hero_player_id]
        mask = self.status_mask[self.game_status]
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        # main_hero = None
        # enemy_hero_hp = None
        # for hero in frame_data["hero_states"]:
        #             if hero["player_id"] == self.main_hero_player_id:
        #                 main_hero = hero
        #             else:
        #                 enemy_hero = hero
        #idle_skill = sum(main_hero['skill_state']['slot_states'][n]['usable'] for n in [1,2,3])
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                balance_rate = 1
                # if frame_data['frameNo']>8000:
                #     balance_rate = 1.2
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value)*balance_rate
                reward_struct.value *= mask[reward_name]
                #(f'{reward_name}:{reward_struct.value}\n')
            
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
                reward_struct.value *= mask[reward_name]
            
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                    reward_struct.value /=50
                # print('id:',self.main_hero_player_id,main_hero['actor_state']['runtime_id'])
                reward_struct.value *= mask[reward_name]
            
            elif reward_name == 'game_status':
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]
            
            elif reward_name == 'attack_tower':
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value# - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                #if_use = reward_struct.cur_frame_value - reward_struct.last_frame_value
                reward_struct.value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_main_calc_frame_map[reward_name].cur_frame_value
                # print(self.main_hero_player_id, reward_struct.value)
                # print("status", self.game_status)
                # print("mask", mask[reward_name])
                # print("----------------------------------")
                if reward_struct.value<0.0005:
                    reward_struct.value=0
                else:reward_struct.value=1

                reward_struct.value *= mask[reward_name]
                
            elif reward_name == 'defend_tower':
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]
            
            elif reward_name == 'approach_tower':
                # main_hero = None
                # for hero in frame_data["hero_states"]:
                #     if hero["player_id"] == self.main_hero_player_id:
                #         main_hero = hero
                # print('last at',main_hero["actor_state"]["runtime_id"],self.m_main_calc_frame_map[reward_name].last_frame_value)
                # print('cur at',main_hero["actor_state"]["runtime_id"],self.m_main_calc_frame_map[reward_name].cur_frame_value)
                if self.m_main_calc_frame_map[reward_name].last_frame_value ==0 or self.m_main_calc_frame_map[reward_name].cur_frame_value==0 :
                    reward_struct.value =0
                else:
                    reward_struct.value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_main_calc_frame_map[reward_name].cur_frame_value
                if reward_struct.value<0:
                    reward_struct.value*=2
                reward_struct.value *= mask[reward_name]
            
            elif reward_name == "retreat":
                #reward_struct.value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_main_calc_frame_map[reward_name].cur_frame_value
                if self.m_main_calc_frame_map[reward_name].last_frame_value ==0 or self.m_main_calc_frame_map[reward_name].cur_frame_value==0 :
                    reward_struct.value =0
                else:
                    reward_struct.value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_main_calc_frame_map[reward_name].cur_frame_value
                if reward_struct.value<0:
                    reward_struct.value*=1.5
                reward_struct.value *= mask[reward_name]

            elif reward_name == "last_hit":
                
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]
            elif reward_name =='enemy_sodiler_pos':
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]
            elif reward_name == 'enemy_sodiler_hp':
                
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                #reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value #- reward_struct.last_frame_value
                reward_struct.value *= mask[reward_name]

            elif reward_name == "kill":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # balance_rate = -1
                # if frame_data['frameNo']>1000 or reward_struct.cur_frame_value >=2 :
                #     balance_rate = 1
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)#*balance_rate
                reward_struct.value *= mask[reward_name]

            elif reward_name == "death":
                balance_rate = 1
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # if frame_data['frameNo']<2000 and (reward_struct.last_frame_value==0 and reward_struct.cur_frame_value==1):
                #     balance_rate*=2
                # else:
                #     balance_rate +=math.sqrt(frame_data['frameNo']/20000)
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)#*balance_rate
                reward_struct.value *= mask[reward_name]
                

            elif reward_name =='heal':
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value# - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                if_use = reward_struct.cur_frame_value - reward_struct.last_frame_value
                
                if  int(if_use):
                    if main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]>0.85:
                        balance_rate -= 0.3+main_hero["actor_state"]["hp"]/main_hero["actor_state"]["max_hp"]
                
                #print(main_hero['buff_state']['buff_marks'])
                else:
                    balance_rate = 0
                check_hp = self.check_hp(self.last_few_frame_hp)
                #print(check_hp)
                if check_hp > 0:
                    balance_rate += self.check_hp(self.last_few_frame_hp)*20

                reward_struct.value = balance_rate
                reward_struct.value *= mask[reward_name]

                
                #print(f'{reward_name}:{reward_struct.value}')

            elif reward_name == "skill_hit_count":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value #- self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value-reward_struct.last_frame_value
                reward_struct.value *= mask[reward_name]
            elif reward_name == "skill_hit_count_bonus":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value #- self.m_enemy_calc_frame_map[reward_name].last_frame_value
                if reward_struct.cur_frame_value==0:
                    reward_struct.value=0
                else:
                    reward_struct.value = reward_struct.cur_frame_value-reward_struct.last_frame_value
                reward_struct.value *= mask[reward_name]
            # elif reward_name == 'hurt_eff':
            #     reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == 'flash':
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]
            # elif reward_name == 'aggressive':
            #     reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == 'forward':
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.value *= mask[reward_name]

            elif reward_name == "HurtToHero":

                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
                reward_struct.value *= mask[reward_name]
                #print(f'{reward_name}:{reward_struct.value}\n')
                

            elif reward_name == "BeHurtByHero":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
                reward_struct.value *= mask[reward_name]
                #print(f'{reward_name}:{reward_struct.value}')
            
            elif reward_name == "HurtToOthers":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value-reward_struct.last_frame_value
                reward_struct.value *= mask[reward_name]
                #print(f'{reward_name}:{reward_struct.value}')
            
            elif reward_name == "tower_hp_point":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value) 
                reward_struct.value *= mask[reward_name]
                # if self.game_status == "attackTower":
                #     print(main_hero["actor_state"]['runtime_id'],'get',self.m_main_calc_frame_map[reward_name].cur_frame_value)
                #print(f'{reward_name}:{reward_struct.value}')
            # elif reward_name == 'skill':
            #     reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name =='strengthen':
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value# - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value #- self.m_enemy_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = reward_struct.cur_frame_value-reward_struct.last_frame_value
                if reward_struct.last_frame_value>=1 and reward_struct.cur_frame_value>=1:
                   reward_struct.value+=1
                reward_struct.value *= mask[reward_name]

            
            # elif reward_name =='self_play':
            #     reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            #     reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value 
            #     reward_struct.value =(reward_struct.cur_frame_value - reward_struct.last_frame_value)
            #     if reward_struct.value>0:
            #         if main_hero['actor_state']['config_id'] ==508 and main_hero['level']>=2:
            #             reward_struct.value=-1 if main_hero['level']<4 else -1.5
            #             if 9000<main_hero['actor_state']['attack_range'] :
            #                 reward_struct.value+=0.5
            #             if main_hero['actor_state']['values']['crit_rate']>3000 and main_hero['level']>=4:
            #                 reward_struct.value+=0.25
            #         elif main_hero['actor_state']['config_id'] ==133:
            #             reward_struct.value -=idle_skill
            #         elif main_hero['actor_state']['config_id'] ==199:
            #             reward_struct.value=-1
            #             if 6900>main_hero['actor_state']['attack_range'] :
            #                 reward_struct.value+=0.5
            #     else:
            #         reward_struct.valuee=0
            
            # elif reward_name == "enemy_Soldiers_hp":
                
            #     # reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
            #     # reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - self.m_enemy_calc_frame_map[reward_name].last_frame_value
            #     #reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
            #     reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            #     reward_struct.value = reward_struct.cur_frame_value
            #     #print('cur_frame_value:',reward_struct.cur_frame_value,'last_frame_value:',reward_struct.last_frame_value)
            #     #print(reward_struct.value)

            elif reward_name == "money":
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)/100
            elif reward_name =='useSkillToHero':
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if self.combat_status:
                    if main_hero['actor_state']['config_id'] ==133:
                        reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value #- self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                        reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value #- self.m_enemy_calc_frame_map[reward_name].last_frame_value
                        reward_struct.value = (reward_struct.cur_frame_value - reward_struct.last_frame_value)
                        if reward_struct.last_frame_value!=0 and reward_struct.value==0:
                            reward_struct.value=-0.2
                else:reward_struct.value = 0



            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value

        # print('#################Current Reward##############')
        # for reward_name in reward_dict.keys():
        #     print(f'{reward_name}:{reward_dict[reward_name]}\n')

        reward_dict["reward_sum"] = reward_sum
        #print(reward_sum)





