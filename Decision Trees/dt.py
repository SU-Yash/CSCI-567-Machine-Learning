import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.parent = None
        self.prune_accuracy = 0
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max =  label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    def split(self):
        features = np.array(self.features)
        number_of_attributes = features[0].size

        if(number_of_attributes != 0):

            max_info_gain = -1
            max_unique_values = 0
            max_attr_number = 0

            #Entropy for root:
            root_branch = []
            root_label_count = []
            for label in np.unique(self.labels):
                root_label_count.append(self.labels.count(label))
            root_branch.append(root_label_count)
            entropy_root = Util.Information_Gain(0, root_branch)
            entropy_root *= -1

            #Split according to attributes
            for attr_number in range(number_of_attributes):
                unique_values = np.unique(features[:,attr_number])
                splits = unique_values.size
                branches = []
                children = []

                np.sort(unique_values)
                for unique_value in unique_values:
                    branch = []
                    branch_feat = []
                    branch_label = []
                    branch_label_count = []
                    zero_label = 0
                    one_label = 0
                    for pos, feature in enumerate(features):
                        if(feature[attr_number] == unique_value):
                            branch_feat.append(np.delete(feature, attr_number))
                            branch_label.append(self.labels[pos])

                    for label in np.unique(self.labels):
                        branch_label_count.append(branch_label.count(label))

                    branches.append(branch_label_count)
                    child = TreeNode(branch_feat, branch_label, np.unique(branch_label).size)
                    child.parent = self
                    children.append(child)

                # Check Information Gain for each attribute
                info_gain = Util.Information_Gain(entropy_root, branches)

                if(info_gain > max_info_gain):
                    max_info_gain = info_gain
                    max_unique_values = unique_values.size
                    max_attr_number = attr_number
                    self.children = children
                    self.dim_split = attr_number
                    self.feature_uniq_split = unique_values

                elif(info_gain == max_info_gain):
                    if(unique_values.size > max_unique_values):
                        max_info_gain = info_gain
                        max_unique_values = unique_values.size
                        max_attr_number = attr_number
                        self.children = children
                        self.dim_split = attr_number
                        self.feature_uniq_split = unique_values

                    elif(unique_values.size == max_unique_values):
                        if(attr_number < max_attr_number):
                            max_info_gain = info_gain
                            max_unique_values = unique_values.size
                            max_attr_number = attr_number
                            self.children = children
                            self.dim_split = attr_number
                            self.feature_uniq_split = unique_values
            if(max_info_gain == 0.0):
                self.children = []
                self.dim_split = None
                self.feature_uniq_split = None
                self.splittable = False
                return

            for child in self.children:
                if(child.splittable is True):
                    child.split()
        else:
            self.splittable = False


    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            child_pos = np.where(self.feature_uniq_split == feature[self.dim_split])
            return self.children[child_pos[0].tolist()[0]].predict(np.delete(feature, self.dim_split))
        else:
            return self.cls_max
