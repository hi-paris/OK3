# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


# All submodules and packages
from OK3 import _classes
from OK3 import _forest
from OK3 import base
from OK3 import kernel
from OK3 import structured_object


# OK3 functions
from OK3._classes import *
from OK3._forest import *
from OK3.base import *
from OK3.kernel import *
from OK3.structured_object import *

__version__ = "0.8.2dev"

__all__ = ['ExtraOK3Regressor', 'OK3Regressor', 'BaseKernelizedOutputTree', 
           'BaseOKForest', 'RandomOKForestRegressor',
           'ExtraOKTreesRegressor', 'RandomOKTreesEmbedding', 'StructuredOutputMixin', 
           'Kernel', 'Mean_Dirac_Kernel', 'Gini_Kernel', 'Linear_Kernel',
           'MSE_Kernel', 'Laplacian_Kernel', 'Gaussian_Kernel',
           'Mallows_Kernel', 'StructuredObject'
          ]
