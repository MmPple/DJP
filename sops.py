import re
import matplotlib.pyplot as plt

# 读取文件中的日志数据
# with open('./TBN_s5e-10_t1e-4.log', 'r') as file:
with open('/home/chenshuo/NewPruning24/Unstructured-Pruning-main/cifar100_log/w_1e-8_s_1e-10_lsum_I.log', 'r') as file:
    log_data = file.read()

# 用正则表达式提取SOPs的值
sops_values = re.findall(r'Avg SOPs: ([\d\.]+) G', log_data)

# 将提取到的值转换为浮点数
sops_values = list(map(float, sops_values))

# 假设两组SOPs交替出现
sops_group_1 = sops_values[0::2]
sops_group_2 = sops_values[1::2]

print(sops_group_2)

# # 绘图
# plt.plot(sops_group_1, label='SOPs Group 1')
# plt.plot(sops_group_2, label='SOPs Group 2')
# plt.xlabel('Index')
# plt.ylabel('SOPs Value')
# plt.title('SOPs Values from Log')
# plt.legend()
# plt.show()
'''
import re

# 定义正则表达式模式
continuous_mask_pattern = r"Test \(continuous mask\) Acc@1:\s(\d+\.\d+)"
test_acc_pattern = r"Test Acc@1:\s(\d+\.\d+)"

# 初始化列表用于存储结果
continuous_mask_accs = []
test_accs = []

# 读取文件
with open('dvscifar10_w_1e-8_s_1e-12_t_5e-4.log', 'r') as file:
    for line in file:
        # 查找并存储Test (continuous mask) Acc@1
        continuous_mask_match = re.search(continuous_mask_pattern, line)
        if continuous_mask_match:
            continuous_mask_accs.append(float(continuous_mask_match.group(1)))
        
        # 查找并存储Test Acc@1
        test_acc_match = re.search(test_acc_pattern, line)
        if test_acc_match:
            test_accs.append(float(test_acc_match.group(1)))

# 将两个列表前后连接
combined_accs = continuous_mask_accs + test_accs

# 打印结果
print(combined_accs)
'''