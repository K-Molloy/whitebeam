# gini tree
# Whitebeam | Kieran Molloy | Lancaster University 2020

"""
This class implements the classification tree in CART 
by inheriting the alpha tree (alpha=2).
"""

from whitebeam.base import AlphaTreeClassifier
import numpy as np

class DecisionTreeClassifier(AlphaTreeClassifier):

    def __init__(self, 
                max_depth=5, 
                min_samples_split=2,
                min_samples_leaf=1):

        AlphaTreeClassifier.__init__(self, 
                        alpha=2.0,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf)


