# test 
# test pt4-2
# test pt2-2
# test pt3-2
# test pt6-2
# test pt8
from torch import linalg as LA
sum_all=[]
for i in range(len(list(model.parameters()))):
    if list(model.parameters())[i].grad!=None:
        sum_all.append(LA.norm(list(model.parameters())[i].grad.cpu()))

sum_all=torch.FloatTensor(sum_all)
print(LA.norm(sum_all))

grad_main=LA.norm(list(model.parameters())[0].grad.cpu())
print(grad_main)

sum_all1=[]
for i in range(len(linear_paras)):
    if linear_paras[i].grad!=None:
        sum_all1.append(LA.norm(linear_paras[i].grad.cpu()))

sum_all1=torch.FloatTensor(sum_all1)
print(LA.norm(sum_all1))

grad_branch=LA.norm(linear_paras[0].grad.cpu())#.cpu().numpy()
print(grad_branch)


for i in range(len(list(model.parameters()))):
    print(LA.norm(list(model.parameters())[i].grad.cpu()))
