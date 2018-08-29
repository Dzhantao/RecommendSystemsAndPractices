import numpy as np


class record():
    def __init__(self, w, r):
        self.weight = w
        self.reason = r

'''
    基于itemCF算法
'''
def ItemSimilarity(train):
    C = dict()
    N = dict()
    W = dict()
    for u ,item_set in train.items():
        for i in item_set:
            N[i] = 0 if i not in N else N[i]
            C[i] = {} if i not in C else C[i]
            N[i] += 1
            for j in item_set:
                C[i][j] = 0 if j not in C[i] else C[i][j]
                if i == j :
                    continue
                C[i][j] += 1

    # 相似度计算
    for i,related_items in C.items():
        W[i] = {} if i not in W else W[i]
        for j , cij in related_items.items():
            if cij != 0 :
                W[i][j] = 0 if j not in W[i] else W[i][j]
                W[i][j] += cij / np.sqrt(N[i] * N[j])
    return W

'''
    基于itemCF算法,相似度进行归一化：
        归一化的好处不仅仅在于增加推荐的准确度，它还可以提高推荐的覆盖率和多样性
'''
def ItemSimilarity_Norm(train):
    C = dict()
    N = dict()
    W = dict()
    W_Norm = dict()
    for u ,item_set in train.items():
        for i in item_set:
            N[i] = 0 if i not in N else N[i]
            C[i] = {} if i not in C else C[i]
            N[i] += 1
            for j in item_set:
                C[i][j] = 0 if j not in C[i] else C[i][j]
                if i == j :
                    continue
                C[i][j] += 1

    # 相似度计算
    for i,related_items in C.items():
        W[i] = {} if i not in W else W[i]
        for j , cij in related_items.items():
            if cij != 0 :
                W[i][j] = 0 if j not in W[i] else W[i][j]
                W[i][j] += cij

    max_similarity = dict()
    for item,item_similarity in W.items():
        for i,similarity in item_similarity.items():
            max_similarity[i] = 0 if i not in max_similarity else max_similarity[i]
            if similarity > max_similarity[i] :
                max_similarity[i] = similarity
    for item, item_similarity in W.items():
        W_Norm[item] = {} if item not in W_Norm else W_Norm[item]
        for j,similarity in item_similarity.items():
            W_Norm[item][j] = 0 if j not in W_Norm[item] else W_Norm[item][j]
            W_Norm[item][j] = W[item][j] / (max_similarity[item] * 1.0)

    return W_Norm


'''
    基于IUF算法的物品相似度: 
        降低活跃用户对相似度的影响；对于过于活跃的用户，可以直接删除兴趣列表
'''
def ItemSimilarity_IUF(train):
    C = dict()
    N = dict()
    W = dict()
    # 构造用户-物品倒排表
    for u ,item_set in train.items():
        for i in item_set:
            N[i] = 0 if i not in N else N[i]
            C[i] = {} if i not in C else C[i]
            N[i] += 1
            for j in item_set:
                C[i][j] = 0 if j not in C[i] else C[i][j]
                if i == j :
                    continue
                C[i][j] += 1 / np.log(1 + len(item_set) * 1.0)

    # 相似度计算
    for i,related_items in C.items():
        W[i] = {} if i not in W else W[i]
        for j , cij in related_items.items():
            if cij != 0 :
                W[i][j] = 0 if j not in W[i] else W[i][j]
                W[i][j] += cij / np.sqrt(N[i] * N[j])
    return W




def Recommand_Norm_withReason(user, user_items, W, K):
    rank = dict()
    ru = user_items[user]
    behavior = 0
    for i, pi in ru.items():
        behavior += pi
        for j, wj in sorted(W[i].items(), key=lambda x: x[1], reverse=True)[:K]:
            if not j in ru:
                rank[j] = record(0, {}) if j not in rank else rank[j]
                rank[j].weight += pi * wj
                rank[j].reason[i] = pi * wj  # 推荐理由
    for j in rank :
        rank[j].weight = rank[j].weight / (behavior * 1.0)
    return rank

def Recommand_withReason(user, user_items, W, K):
    rank = dict()
    ru = user_items[user]
    for i, pi in ru.items():
        for j, wj in sorted(W[i].items(), key=lambda x: x[1], reverse=True)[:K]:
            if not j in ru:
                rank[j] = record(0, {}) if j not in rank else rank[j]
                rank[j].weight += pi * wj
                rank[j].reason[i] = pi * wj  # 推荐理由
    return rank
def main():
    user_items = {"Allen": {'a': 1, 'b': 1, 'd': 1},
                  "Ben": {'b': 1, 'c': 1, 'e': 1},
                  "Copper": {'c': 1, 'd': 1},
                  "Denny": {'b': 1, 'c': 1, 'd': 1},
                  "Eric": {'a': 1, 'd': 1}}

    # 得到相似度集合
    W = ItemSimilarity_Norm(user_items)
   # print(W)
    W2 = ItemSimilarity_IUF(user_items)


    # 相似度集合转矩阵
    # result =  np.zeros((5,5))
    # items = ['a','b','c','d','e']
    # for i,reated_items in W.items():
    #     for j, w in reated_items.items():
    #         result[items.index(i)][items.index(j)] = w

    rank = Recommand_Norm_withReason('Allen', user_items,W, K=3)
    rank2 = Recommand_withReason('Allen', user_items, W2, K=3)
    for recommend_item, record in rank.items():
        print('recommend ', recommend_item, record.weight)
        print('computed from ', [(k, v) for k, v in record.reason.items()])
    for recommend_item, record in rank2.items():
        print('recommend2 ', recommend_item, record.weight)
        print('computed2 from ', [(k, v) for k, v in record.reason.items()])
if __name__ == '__main__':
    main()