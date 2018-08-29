import numpy as np


'''
    基于UserCF算法
'''

def UserSimilarity(user_Items):
    # 构造物品-用户相似度
    item_users = dict()
    for user,item_set in user_Items.items():
        for k in item_set:
            if k not in item_users :
                item_users[k] = set()
            item_users[k].add(user)

    # 有共同浏览记录的用户的重合度
    C = dict()  # C(u,v)
    N = dict()  # N(u),N(v)

    for item,user_set in item_users.items():
        for u in user_set:
            N[u] = 0 if u not in N else N[u]
            C[u] = dict() if u not in C else C[u]
            N[u] += 1
            for v in user_set:
                C[u][v] = 0 if v not in C[u] else C[u][v]
                if u != v :
                    C[u][v] += 1


    # 计算最终相似度矩阵
    W = dict()
    for u,reated_user in C.items():
        W[u] = dict() if u not in W else W[u]
        for v ,cuv in reated_user.items():
            if cuv != 0:
                W[u][v] = 0 if v not in W[u] else W[u][v]
                W[u][v] = cuv / np.sqrt(N[u] * N[v])

    return W



'''
    基于User-IIF算法
'''
def UserSimilarity_IIF(user_Items):
    item_users = dict()
    for u,items in user_Items.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)

    C = dict()
    N = dict()
    for i ,users in item_users.items():
        for u in users:
            N[u] = 0 if u not in N else N[u]
            C[u] = dict() if u not in C else C[u]
            N[u] += 1
            for v in users:
                C[u][v] = 0 if v not in C[u] else C[u][v]
                if u != v :
                    C[u][v] += 1 / np.log(1 + len(users))

    W = dict()
    for u,related_users in C.items():
        W[u] = dict() if u not in W else W[u]
        for v,cuv in related_users.items():
            W[u][v] = cuv / np.sqrt(N[u] * N[v])

    return W




# 推荐给用户
def Recommand(user,user_items,W,k):
    rank = dict()
    interacted_items = user_items[user]
    for v,wuv in sorted(W[user].items(),key= lambda x:x[1],reverse=True)[:k]:
        for i,rvi in user_items[v].items():
            if i not in interacted_items:
                rank[i] = 0 if i not in rank else rank[i]
                rank[i] += wuv * rvi
    return rank



user_items = {"Allen": {'a':1, 'b':1, 'd':1},
              "Ben":   {'a':1, 'c':1},
              "Copper":{'b':1, 'e':1},
              "Denny": {'c':1, 'd':1, 'e':1}}


W1 = UserSimilarity(user_items)
W2 = UserSimilarity_IIF(user_items)


rank = Recommand("Allen", user_items, W1, k=3)
print(rank)
rank = Recommand("Allen", user_items, W2, k=3)
print(rank)