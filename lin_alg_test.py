import linearalgebra as la
from fractions import Fraction

"""
This file is for testing the functions, class, and class methods.
Please understand that multiplication and division of float data types (decimals) may cause
slight computational errors (such as a difference of 0.00000001 from a correct result)
due to the computer's system using bits for computations.
"""

# (A) squared matrices
matA1 = la.Matrix([[-6, 2],
                   [-1, 2]])

matA2 = la.Matrix([[4, 0, 0, 0],
                   [Fraction(3, 4), 4.33, 0, 0],
                   [0, 2.2, 3.43, 0],
                   [4, 14.3, 7, 3]])

    # (A3) non-invertible matrix
matA3 = la.Matrix([[33, 2, 6, 4.3],
                   [0, 2.1, 3, 1],
                   [0, 0, 0, 43],
                   [0, 0, 0, 4.33]])

matA4 = la.Matrix([[2, 1],
                   [1, 0]])

# (B) unsquared matrix
matB1 = la.Matrix([[10, 6],
                   [2, 3],
                   [2, 5]])

matB2 = la.Matrix([[1, 2],
                   [3.4, 2],
                   [1, 2]])

matB3 = la.Matrix([[1, 0],
                   [-2, 2],
                   [-5, 4]])

# (C) unsquared matrix
matC1 = la.Matrix([[1, 5, -3],
                   [-7, 1, 3]])

# (D) linearly dependent matrix
matD1 = la.Matrix([[0, 0, 0, 3],
                   [0, 0, 2, 4],
                   [0, 1, 3, 8]])

matD2 = la.Matrix([[1, 3, 3, 2, -9],
                   [-2, -2, 2, -8, 2],
                   [3, 4, -1, 11, -8],
                   [2, 3, 0, 7, 1]])

# (E) matrix with Fractions and floats
matE1 = la.Matrix([[Fraction(1, 3), 12.6, 0],
                   [2.7, 5.3, 34]])

# (F) one line matrix
matF1 = la.Matrix([[4, -1, -1, 3]])

if __name__ == '__main__':
    print(f'columns 1: {matB1.get_self_columns()}\n')   # method
    print(f'columns 2: {la.get_columns(matB1)}\n')   # function

    print(f'REF:\n{matA1.getREF()}\n')

    print(f'RREF1:\n{matB1.getRREF()}\n')
    print(f'RREF2:\n{matC1.getRREF()}\n')
    print(f'RREF3:\n{matE1.getRREF()}\n')

    print(f'determinant1:\n{matA1.getDeterminant()}\n')
    print(f'determinant2:\n{matA2.getDeterminant()}\n')
    print(f'determinant3:\n{matA3.getDeterminant()}\n')

    print(f'inverse1:\n{matA1.getInverse()}\n')
    # raises error because not invertible
    # print(f'inverse2:\n{matA3.getInverse()}\n')

    print(f'transpose1:\n{matB1.getTranspose()}\n')
    print(f'transpose2:\n{matC1.getTranspose()}\n')

    print(f'scaler-matrix-multipl 1-1:\n{matA1.getScalarMultip(2)}\n')  # method
    print(f'scaler-matrix-multipl 1-2:\n{la.matrix_scalar_multip(matA1, 2)}\n')     # function
    print(f'scaler-matrix-multipl 2-1:\n{matE1.getScalarMultip(1.5)}\n')    # method
    print(f'scaler-matrix-multipl 2-2:\n{la.matrix_scalar_multip(matE1, 1.5)}\n')   # function

    print(f'matrix-addition 1-1:\n{matA1.getMatrixAdd_Sub(matA4, add=True)}\n')     # method
    print(f'matrix-addition 1-2:\n{la.matrix_add_sub(matA1, matA4, add=True)}\n')   # function
    print(f'matrix-subtraction 1-1:\n{matB1.getMatrixAdd_Sub(matB2, add=False)}\n')     # method
    print(f'matrix-subtraction:\n{la.matrix_add_sub(matB1, matB2, add=False)}\n')   # function
    # raises error because not same dimensions
    #   print(f'matrix-addition/subtraction 3-1:\n{matA1.getMatrixAdd_Sub(matB1, add=True)}\n')    # method
    #   print(f'matrix-addition/subtraction 3-2:\n{la.matrix_add_sub(matA1, matB1, add=True)}\n')    # function

    print(f'matrix-vector-multipl 1-1:\n{matB2.getVectorMultip([2, 1])}\n')  # method
    print(f'matrix-vector-multipl 1-2:\n{la.matrix_vector_multip(matB2, [2, 1])}\n')  # function
    # raises error because not same dimensions
    # print(f'matrix-vector-multipl 2-1:\n{matA1.getVectorMultip([2, 3, 1])}\n')  # method
    # print(f'matrix-vector-multipl 2-2:\n{la.matrix_vector_multip(matA1, [2, 3, 1])}\n')  # function

    print(f'matrix-matrix-multipl 1-1:\n{matB2.getMatrixMultip(matC1)}\n')  # method
    print(f'matrix-matrix-multipl 1-2:\n{la.matrix_multip(matB2, matC1)}\n')  # function
    # raises error because of incompatible dimensions
    # print(f'matrix-matrix-multipl 2-1:\n{matA1.getMatrixMultip(matB2)}\n')  # method
    # print(f'matrix-matrix-multipl 2-2:\n{la.matrix_multip(matA1, matB2)}\n')  # function

    print(f'scaler-vector-multipl:\n{la.vect_scalar_multip([1, 2, 3], 1.5)}\n')

    print(f'vector-addition:\n{la.vector_add_sub([1, 2, 3], [1, 2.5, 2], add=True)}\n')
    print(f'vector-subtraction:\n{la.vector_add_sub([1, 2, 3], [1, 2.5, 2], add=False)}\n')

    print(f'transform 1-1:\n{matB3.getTransform([5, 6])}\n')  # method
    print(f'transform 1-2:\n{la.transform(matB3, [5, 6])}\n')  # function
    # raises error because of incompatible dimensions
    # print(f'transform 2-1:\n{matB3.getTransform([5, 6, 2])}\n')  # method
    # print(f'transform 2-2:\n{la.transform(matB3, [5, 6, 2])}\n')  # function

    print(matD2.getRREF())
    print(f'column-space-basis:\n{matD2.getColBasis()}\n')

    print(f'vectors:\n{matB2.getVectors()}\n')

    print(matD2.getRREF())
    print(f'Linearly dependent (False if linearly dependent):\n{matD2.isLinIndependent()}\n')
    print(matB2.getRREF())
    print(f'Linearly independent (True if linearly independent):\n{matB2.isLinIndependent()}\n')
    print(matA3.getRREF())
    print(f'Linearly dependent (False if linearly dependent):\n{matA3.isLinIndependent()}\n')
    print(matA1.getRREF())
    print(f'Linearly independent (True if linearly independent):\n{matA1.isLinIndependent()}\n')

    print(la.Matrix([[1, -1, 1], [-2, 2, -2]]).getRREF())
    print(f'consistent (True if consistent):\n{la.Matrix([[1, -1], [-2, 2]]).isConsistent(vector_b=[1, -2])}\n')
    print(la.Matrix([[3, 4, 5], [6, 8, 7]]).getRREF())
    print(f'inonsistent (False if inconsistent):\n{la.Matrix([[3, 4], [6, 8]]).isConsistent(vector_b=[5, 7])}\n')

    print(la.Matrix([[3, 2], [-1, 1], [1, -3]]).getRREF(addition=[6, -2, 2]))
    print(f'linear combination (True if linear combination):\n{la.Matrix([[3, 2], [-1, 1], [1, -3]]).isLinComb([6, -2, 2])}\n')

    print(matA1.getRREF(addition=[2, 1]))
    print(f'linear combination (True if linear combination):\n{matA1.isLinComb([2, 1])}\n')
    print(la.Matrix([[3, 5], [1, 0], [2, 1]]).getRREF(addition=[8, 0, 3]))
    print(f'linear combination (False if not linear combination):\n{la.Matrix([[3, 5], [1, 0], [2, 1]]).isLinComb([8, 0, 3])}\n')

    print(f'dimensions:\n{matF1.getDimensions()}\n')

    print(f'eigenvector (True if eigenvector):\n{la.Matrix([[1, 6], [5, 2]]).isEigenvector([6, -5])}\n')
    print(f'eigenvector (True if eigenvector):\n{la.Matrix([[3, 6, 7],[3, 3, 7], [5, 6, 5]]).isEigenvector([1, -2, 1])}\n')
    print(f'eigenvector (False if not eigenvector):\n{la.Matrix([[1, 6], [5, 2]]).isEigenvector([3, -2])}\n')

    print(f'eigenvalue (True if eigenvalue):\n{la.Matrix([[1, 6], [5, 2]]).isEigenvalue(7)}\n')
    print(f'eigenvalue (False if not eigenvalue):\n{la.Matrix([[1, 6], [5, 2]]).isEigenvalue(6)}\n')
