import numpy as np


def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    total_points = np.sum(branches)
    conditional_entropy = 0
    for branch in branches:
        total_branch_points = np.sum(branch)
        branch_entropy = 0
        for point in branch:
            if (point != 0 and total_branch_points != 0):
                branch_entropy = branch_entropy + (-point/total_branch_points)*np.log2(point/total_branch_points)
        conditional_entropy = conditional_entropy + total_branch_points/total_points * branch_entropy
    return S - conditional_entropy

def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    def accuracy(y_pred, y_test):
        import pdb 
        #pdb.set_trace()
        right_count = 0
        for i in range(len(y_pred)):
            if(y_pred[i] == y_test[i]):
                right_count = right_count + 1
        return right_count/len(y_pred) 

    def traverse(node):

        if node.splittable:
            for child in node.children:
                traverse(child)

        else:
            y_pred = decisionTree.predict(X_test)
            old_accuracy = accuracy(y_pred, y_test)
            old_decisionTree = decisionTree

            node_children = node.parent.children
            node_splittable = node.parent.splittable
            node_dim_split = node.parent.dim_split
            node_feature_uniq_split = node.parent.feature_uniq_split
            
            node.parent.children = []
            node.parent.splittable = False
            node.parent.dim_split = None
            node.parent.feature_uniq_split = None

            new_y_pred = decisionTree.predict(X_test)
            new_accuracy = accuracy(new_y_pred, y_test)
            if(new_accuracy <= old_accuracy):
                node.parent.children = node_children
                node.parent.splittable = node_splittable
                node.parent.dim_split = node_dim_split
                node.parent.feature_uniq_split = node_feature_uniq_split
        return
    decisionTree = decisionTree
    traverse(decisionTree.root_node)
    return decisionTree

def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    def accuracy(y_pred, y_test):
        import pdb 
        #pdb.set_trace()
        right_count = 0
        for i in range(len(y_pred)):
            if(y_pred[i] == y_test[i]):
                right_count = right_count + 1
        return right_count/len(y_pred)

    def traverse(node):

        if node.splittable:
            for child in node.children:
                traverse(child)
            prune(node)

        else:
            prune(node.parent)

        return

    def prune(node):
        y_pred = decisionTree.predict(X_test)
        old_accuracy = accuracy(y_pred, y_test)
        old_decisionTree = decisionTree

        node_children = node.children
        node_splittable = node.splittable
        node_dim_split = node.dim_split
        node_feature_uniq_split = node.feature_uniq_split
        
        node.children = []
        node.splittable = False
        node.dim_split = None
        node.feature_uniq_split = None

        new_y_pred = decisionTree.predict(X_test)
        new_accuracy = accuracy(new_y_pred, y_test)
        if(new_accuracy <= old_accuracy):
            node.prune_accuracy = new_accuracy
            node.children = node_children
            node.splittable = node_splittable
            node.dim_split = node_dim_split
            node.feature_uniq_split = node_feature_uniq_split

    traverse(decisionTree.root_node)
    return decisionTree


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    import pdb
    #pdb.set_trace()
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
