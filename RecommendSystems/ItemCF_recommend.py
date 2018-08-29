import numpy as np

'''
    基于itemCF算法
'''
class record():
    def __init__(self, w, r):
        self.weight = w
        self.reason = r
def ItemSimilarity(item_user):
    C = dict()
    N = dict()
    W = dict()
    # 构造用户-物品倒排表
    user_items = dict()
    for item,user_set in item_user.items():
        for k in user_set:
            if k not in user_items:
                user_items[k] = set()
            user_items[k].add(item)
    for u ,item_set in user_items.items():
        for i in item_set:
            N[i] = 0 if i not in N else N[i]
            C[i] = {} if i not in C else C[i]
            N[i] += 1
            for j in item_set:
                C[i][j] = 0 if j not in C[i] else C[i][j]
                if i == j :
                    continue
                C[i][j] += 1
    print(C)
    # 相似度计算
    for i,related_items in C.items():
        W[i] = {} if i not in W else W[i]
        for j , cij in related_items.items():
            if cij != 0 :
                W[i][j] = 0 if j not in W[i] else W[i][j]
                W[i][j] += cij / np.sqrt(N[i] * N[j])
    return W
def Recommand_withReason(user, user_items, W, K):
    rank = dict()
    ru = user_items[user]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=lambda x: x[1], reverse=True)[:K]:
            if not j in ru:
                rank[j] = record(0, {}) if j not in rank else rank[j]
                rank[j].weight += pi * wj
                rank[j].reason[i] = pi * wj
    return rank
def main():
    user_items = {"Allen": {'a': 1, 'b': 1, 'd': 1},
                  "Ben": {'b': 1, 'c': 1, 'e': 1},
                  "Copper": {'c': 1, 'd': 1},
                  "Denny": {'b': 1, 'c': 1, 'd': 1},
                  "Eric": {'a': 1, 'd': 1}}
    item_users = {'a': {"Allen", "Eric"},
                  'b': {'Allen', 'Ben', 'Denny'},
                  'c': {'Ben', 'Copper', 'Denny'},
                  'd': {'Allen', 'Copper', 'Denny', "Eric"},
                  'e': {'Ben'}}
    # 得到相似度集合
    W = ItemSimilarity(item_users)
    #print(W)

    # 相似度集合转矩阵
    result =  np.zeros((5,5))
    items = ['a','b','c','d','e']
    for i,reated_items in W.items():
        for j, w in reated_items.items():
            result[items.index(i)][items.index(j)] = w

   # print(result)
    rank = Recommand_withReason('Allen', user_items,W, K=3)
    for recommend_item, record in rank.items():
        print('recommend ', recommend_item, record.weight)
        print('computed from ', [(k, v) for k, v in record.reason.items()])
if __name__ == '__main__':
    main()