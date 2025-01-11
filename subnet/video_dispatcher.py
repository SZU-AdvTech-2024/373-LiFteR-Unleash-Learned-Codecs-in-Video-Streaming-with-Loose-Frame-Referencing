import math
from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.parent=None
        self.left = None
        self.right = None

# 创建树的函数，考虑树的最大高度
def create_tree(N):
    lst=[i for i in range(N)]
    max_depth = math.ceil(math.log2(N + 1))  # 最大高度
    root = TreeNode(lst[0])
    visited = [False] * N  # 标记节点是否已经加入树
    visited[0] = True  # 根节点已经加入
    
    out_dict=dict.fromkeys(lst) # 记录未加入图中的节点
    del out_dict[lst[0]]


    # 计算树的高度
    def get_tree_height(node):
        if not node:
            return 0
        left_height = get_tree_height(node.left)
        right_height = get_tree_height(node.right)
        return max(left_height, right_height) + 1

    def get_closest_node(node):
        # 找到距离当前节点最近的未加入树中的节点
        closest_idx = None
        closest_dist = float('inf')

        for key,val in out_dict.items():
            dist=abs(key-node.val)
            if(dist<closest_dist):
                closest_dist=dist
                closest_idx=key
        return closest_idx

    # 先序遍历的递归函数
    def preorder_traverse(node,depth):

        # 判断当前树的高度是否已经超过最大高度
        if depth >= max_depth:
            return

        # 将新节点添加到当前节点的子树中
        if node.left is None:
            if len(out_dict)==0:  # 如果所有节点都已经加入树
                return
            closest_idx=get_closest_node(node)
            new_node = TreeNode(closest_idx)
            node.left = new_node
            del out_dict[closest_idx]
            new_node.parent=node
            preorder_traverse(node.left,depth+1)  # 先遍历左子树
        
        
        if node.right is None:
            if len(out_dict)==0:  # 如果所有节点都已经加入树
                return
            closest_idx=get_closest_node(node)
            new_node = TreeNode(closest_idx)
            node.right = new_node
            del out_dict[closest_idx]
            new_node.parent=node
            preorder_traverse(node.right,depth+1)  # 先遍历左子树

    preorder_traverse(root, 1)
    return root

# 获取树的层序遍历结果，返回每一层的节点值
def level_order_traversal(root):
    if not root:
        return []
    
    levels = []  # 用于存储树的每一层节点值
    vals=[]
    queue = deque([root])  # 使用队列存储待访问的节点
    while queue:
        level_size = len(queue)  # 当前层的节点数
        level_nodes = []  # 存储当前层的节点值
        nodes_val=[]
        for i in range(level_size):
            node = queue.popleft()  # 取出队列中的第一个节点
            level_nodes.append(node)  # 添加当前节点值
            nodes_val.append(node.val)
            
            # 如果当前节点有左子节点，将左子节点加入队列
            if node.left:
                queue.append(node.left)
            
            # 如果当前节点有右子节点，将右子节点加入队列
            if node.right:
                queue.append(node.right)

        levels.append(level_nodes)  # 将当前层的节点值加入levels列表
        vals.append(nodes_val)  

    print("level_order_traversal",vals)
    return levels


if __name__ == '__main__':
    # 测试
    GOP=7
    print([i for i in range(GOP)])
    root = create_tree(GOP)
    # 获取树的层序遍历结果
    levels = level_order_traversal(root)
    # 输出节点值
    print(levels[1][1].parent.val)
