
"""
Author: Hiroshi Nonaka
This python file provides Matrix class, which contains basic linear algebraic functions
and other functinos independent of matrices' self operatinos.

Class
-----
Matrix
    A class object that contains a matrix instances and basic linear algebraic functions.
    For more information, call Matrix class.

Functions
---------
get_columns
    Returns the columns of the given matrix.

convert_fracs
    Converts all the Fraction objects equivalent to an integer into the corresponding int type.
    Ex) 2/1: Fraction(2, 1) -> 2: int

vec_scalar_multip
    Computes constant-vector multiplication.

vector_add_sub
    Computes the vector addition. Vector 1 and 2 should have the same dimensions.

matrix_scalar_multip
    Computes the constant-matrix multiplication.

matrix_vector_multip
    Computes vector-matrix multiplication between the given matrix and vector.

matrix_multip
    Computes matrix multiplication. Matrix 1 is to be multiplied by matrix 2. Ex) (matrix 1) * (matrix 2)

matrix_add_sub
    Computes matrix addition.

transform
    Computes the transformation of the given vector based on the given standard matrix.

"""

import math
from copy import deepcopy
from fractions import Fraction
import numpy as np


class Matrix:
    """
    A matrix that holds basic linear algebraic functions.

    Attributes
    ----------
    matrix: list
        A 2-dimensional list representing a matrix.
    rref_matrix: Matrix
        The RREF of the matrix.
    ref_matrix: Matrix
        The REF of the matrix
    transpose: Matrix
        The transpose of the matrix.
    inverse: Matrix
        The inverse of the matrix.
    identity: Matrix
        The n x n identity matrix of the self matrix if the matrix is squared.
    """
    def __init__(self, matrix: list):
        """
        Parameters
        ----------
        matrix : list
            a 2-dimensional matrix whose level-1 lists represents a row.

        Examples
        --------
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        [[1.2, 5, 3.6], [Fraction(3, 4), 4.5, 1], [3, 2, 8]]

        Raises
        ------
        ValueError
            Raises ValueError if the lengths of the rows are not unified.
        """

        if any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError("Lengths of rows are not the same")
        self.matrix = list(matrix)
        self.rref_matrix = None
        self.ref_matrix = None
        self.transpose = None
        self.inverse = None
        self.identity = None
        # create identity matrix if the matrix is square
        if len(matrix) == len(matrix[0]):
            n = len(matrix)
            self.identity = [[0 for j in range(n)] for i in range(n)]
            for i in range(n):
                self.identity[i][i] = 1

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            A formatted matrix without commas and square brackets.
        """
        n = self.getlongest()
        row = list(" ".join([f'{str(e):>{n}}' for e in row]) for row in self.matrix)
        return "\n".join([f"|{i}|" for i in row])

    def __getitem__(self, key):
        return self.matrix[key]

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __len__(self) -> int:
        return len(self.matrix)

    def getlongest(self) -> int:
        """
        Identifies the number of characters of the longest element in the self matrix.

        Returns
        -------
        int
            the character count of the longest element.
        """
        a = []
        for row in self.matrix:
            a.extend(row)
        a = [len(str(e)) for e in a]
        return max(a)

    def get_self_columns(self) -> list:
        """
        Returns the columns of the self matrix by converting rows into columns.

        Returns
        -------
        columns: list
            the columns of the matrix.
        """
        matrix = deepcopy(self.matrix)
        n = len(matrix[0])
        columns = []
        for i in range(n):
            columns.append([])
            for r in matrix:
                columns[i].append(r[i])
        return columns

    def allzero(self, row):
        """
        Checks if all the elements of the given row is zero.

        Parameters
        ----------
        row: list

        Returns
        -------
        bool
            True if elements are all zero and False if there is a non-zero element.
        """
        return all(x == 0 for x in row)

    def getREF(self, addition: list = None) -> "Matrix":
        """
        Computes the Row Echelon Form of the self matrix.

        Parameters
        ----------
        addition: list, optional
            Any additional matrices/lists added to the right side of the self matrix.
            The integrated matrix is computed as one matrix.
            Used for Ax = 0 or the inverse calculation etc...

        Returns
        -------
        REF: Matrix
            The Row Echelon Form of the initial matrix.
        """
        matrix = deepcopy(self.matrix)
        # chekcs errors about "addition"
        if addition:
            npaddition = np.array(addition)
            if npaddition.ndim == 1 and len(addition) != len(matrix):
                # if the additional list is 1 dimension(vector)
                # and the dimension is different from that of matrix
                raise ValueError(
                    f'Dimension of vector should match the dimension of matrix:'
                    f'{len(matrix)} x {len(matrix[0])}')
            elif npaddition.ndim == 2 and len(addition) != len(matrix):
                # if the additional list is 2 dimension(matrix)
                # and the dimension is different from that of the original matrix
                raise ValueError(
                    f'Dimension of added matrix should match the dimension of original matrix:'
                    f'{len(matrix)} x {len(matrix[0])}')
            elif npaddition.ndim > 2:
                raise ValueError(f'Invalid list depth:{npaddition.ndim}')
            # merge the additional list/matrix to the original matrix
            for i in range(len(addition)):
                matrix[i]: list
                if npaddition.ndim == 1:
                    matrix[i].append(addition[i])
                elif npaddition.ndim == 2:
                    matrix[i] += addition[i]

        n_c = len(matrix[0])
        n_r = len(matrix)
        # put all-zero rows at the bottom of the matrix and sort based on the number of non-zero entries
        matrix.sort(key=lambda x: x.count(0))
        # convert all the floats into Fraction type so that floats can be used as a denominator of Fractions
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if type(matrix[i][j]) == float:
                    matrix[i][j] = Fraction(str(matrix[i][j]))

        for i in range(n_r):
            # Select a nonzero entry in the pivot column as a pivot.
            maxRow = i
            if self.allzero(matrix[i]):
                continue
            pivot = matrix[maxRow][i]
            for idx in range(i + 1, n_r):
                ent = matrix[idx][i]
                if pivot == 0 and abs(ent) > abs(pivot):
                    pivot = ent
                    maxRow = idx
            if matrix[i][maxRow] == 0:
                continue  # Matrix is singular
            # If necessary, interchange rows to move this entry into the pivot position.
            if i != maxRow:
                matrix[maxRow], matrix[i] = matrix[i], matrix[maxRow]
            # Use row replacement operations to create zeros in all positions below the pivot.
            for k in range(i + 1, n_r):
                c = Fraction(matrix[k][i], matrix[i][i])
                for j in range(i + 1, n_c):
                    matrix[k][j] -= matrix[i][j] * c
                matrix[k][i] = 0

        # row-reduces rows with the same pivot positions
        top = 0
        for i in range(1, len(matrix)):
            # find the leading entry
            le_idx = None
            top_le_idx = None
            for j in range(len(matrix[i])):
                if le_idx is None and matrix[i][j] != 0:
                    le_idx = j
                if top_le_idx is None and matrix[top][j] != 0:
                    top_le_idx = j
            # if the positions of leading entries are the same,
            # and if le_idx nor top_le_idx is not None
            if le_idx == top_le_idx and le_idx is not None and top_le_idx is not None:
                c = Fraction(matrix[top][top_le_idx], matrix[i][le_idx])
                for j in range(len(matrix[i])):
                    matrix[i][j] = matrix[top][j] - matrix[i][j] * c
                matrix[i][le_idx] = 0
            # else, move on to the new row as a top row
            else:
                top += 1

        ref = convert_fracs(matrix)
        if addition is None:
            self.ref_matrix = ref
        return Matrix(ref)

    def getRREF(self, addition: list = None) -> "Matrix":
        """
        Computes the Row Reduced Echelon Form of the self matrix

        Parameters
        ----------
        addition: list, optional
            Any additional matrices/lists added to the right side of the self matrix.
            The integrated matrix is computed as one matrix.
            Used for Ax = 0 or the inverse calculation etc...

        Returns
        -------
        REF: Matrix
            The Row Reduced Echelon Form of the initial matrix.
        """
        ref = self.getREF(addition)
        les_idx = []
        # identify pivot positions
        for i in range(len(ref)):
            if self.allzero(ref[i]) is True:
                continue  # Matrix is singular
            pointer = 0
            while ref[i][pointer] == 0:
                pointer += 1
            les_idx.append((i, pointer))
        les_idx.sort(reverse=True)
        for idxi, idxj in les_idx:
            # devide the current row so that the leading entry can be 1
            ref[idxi] = [Fraction(e, ref[idxi][idxj]) for e in ref[idxi]]
            # for each row above the current row
            for i in range(idxi):
                c = ref[i][idxj]
                newline = []
                # create a new row by subtracting the row above by the current row times factor
                for k, m in zip(ref[i], ref[idxi]):
                    newline.append(k - m * c)
                ref[i] = newline
        # namely, ref = RREF with fractinos
        rref = convert_fracs(ref)
        if addition is None:
            self.rref_matrix = rref
        return Matrix(rref)

    def getDeterminant(self, matrix=None) -> int:
        """
        Recursively computes the determinant of the self matrix with the cofactor expansion approach.

        Parameters
        ----------
        matrix: Matrix or list, optional
            For the recursion purpose. A smaller matrix of the initial matrix.

        Raises
        ------
        ValueError
            Raises ValueError if the self matrix is not squared.

        Returns
        -------
        determinant: int
            The determinant of the self matrix.
        """
        if matrix is None:
            matrix = deepcopy(self.matrix)
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix is not square")
        det = 0
        rowkey = 0
        # BASE CASES: if matrix is 2 x 2, return the determinant
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        for i in range(len(matrix)):
            new_matrix = deepcopy(matrix)
            # remove the first row
            new_matrix.pop(rowkey)
            # remove the column that the factor element belongs to
            for r in range(len(new_matrix)):
                new_matrix[r].pop(i)
            # the formula to compute the determinant
            det += (-1) ** (rowkey + i) * matrix[rowkey][i] * self.getDeterminant(matrix=new_matrix)
        return det

    def getInverse(self) -> "Matrix":
        """
        Computes the inverse of the self matrix by using the row reduction approach.

        Raises
        ------
        ValueError
            Raises ValueError if the determinant of the self matrix is 0.

        Returns
        -------
        inverse: Matrix
            The inverse matrix of the self matrix.
        """
        if self.getDeterminant() == 0:
            raise ValueError('This matrix is not invertible')
        n = len(self.matrix)
        in_inv = self.getRREF(addition=self.identity)
        inv = [row[n:] for row in in_inv]
        self.inverse = inv
        return Matrix(inv)

    def getTranspose(self) -> "Matrix":
        """
        Computes the transpose of the self matrix.

        Returns
        -------
        transpose: Matrix
            The transpose of the self matrix.
        """
        # number of colomuns of the original matrix  = number of rows of the transpose
        transpose = [[] for i in range(len(self.matrix[0]))]
        for row in self.matrix:
            count = 0
            for e in row:
                transpose[count].append(e)
                count += 1
        self.transpose = transpose
        return Matrix(transpose)

    def getScalarMultip(self, constant) -> "Matrix":
        """
        Computes the scalar-matrix multiplication between the self matrix and a given scalar.

        Parameters
        ----------
        scalar: numeric types
            A scalar to be multiplied with the self matrix.

        Returns
        -------
        Matrix
            The multiplied version of the self matrix with the given scalar.
        """
        return Matrix(np.array(self.matrix) * constant)

    def getMatrixAdd_Sub(self, matrix: "Matrix", add=True) -> "Matrix":
        """
        Computes the matrix addition or subtraction with another matrix.

        Parameters
        ----------
        matrix: Matrix
            Another matrix to be added to or subtracted from the self matrix.
        add: bool, default=True
            True for computing the addition and False for computing the subtraction.

        Raises
        ------
        ValueError
            Raise ValueError if the self matrix's dimensions are distinct from those of the given matrix.

        Returns
        -------
        Matrix
            The computed version of the self matrix with the given matrix.
        """
        if self.getDimensions() != matrix.getDimensions():
            raise ValueError("Dimensions are distinct")
        new = deepcopy(self.matrix)
        for i in range(len(new)):
            for j in range(len(new[0])):
                if add is True:
                    new[i][j] += matrix[i][j]
                else:
                    new[i][j] -= matrix[i][j]
        return Matrix(new)

    def getVectorMultip(self, vector: list):
        """
        Computes the vector-matrix multiplication between the self matrix and the given vector

        Parameters
        ----------
        vector: list
            A vector to be multiplied to the self matrix.

        Returns
        -------
        Matrix
            The multiplied version of the self matrix with the given vector.
        """
        # if the dimension of given vector is different from the number of columns of given matrix
        if len(self.matrix[0]) != len(vector):
            raise ValueError('Vector should be the same n dimensions as an m X n matrix')
        columns = self.get_self_columns()
        # update columns by multiplying each constant from the vector with a corresponding column
        for i in range(len(vector)):
            columns[i] = vect_scalar_multip(columns[i], vector[i])
        # switch columns into rows
        rows = get_columns(columns)
        # sum up the numbers in the same row
        for i in range(len(rows)):
            rows[i] = sum(rows[i])
        # return multiplied vector
        return rows

    def getMatrixMultip(self, matrix: "Matrix") -> "Matrix":
        """
        Computes the matrix multiplication between the self matrix and the given matrix.
        The order of the multiplication is (self matrix) x (given matrix).

        Parameters
        ----------
        matrix: Matrix

        Raises
        ------
        ValueError
            Raises ValueError if the number of columns of the self matrix is distinct from
            the number of columns of the rows of the given matrix.

        Returns
        -------
        Matrix
            The multiplied version of the self matrix.
        """
        matrix1 = deepcopy(self.matrix)
        matrix2 = matrix
        if len(matrix1[0]) != len(matrix2):
            raise ValueError('Matrix should be an n X k matrix when multiplied to the m X n matrix')
        columns = matrix.get_self_columns()
        # for each matrix, computes matrix-vector (of the second matrix) multiplication
        for j in range(len(columns)):
            matrix1[j] = self.getVectorMultip(columns[j])
        return Matrix(get_columns(matrix1))

    def getTransform(self, vector:list) -> list:
        """
        Computes the transformation of the given vector based on the self matrix as the standard matrix.

        Parameters
        ----------
        vector: list
            A vector to be transformed.

        Returns
        -------
        list
            A transformed vector based on self matrix.
        """
        return self.getVectorMultip(vector)

    def getColBasis(self):
        """
        Returns the columns space basis for self-matrix.

        Returns
        -------
        col_basis: list
            List of column space basis.
        """
        rref = self.getRREF()
        pivots = []
        for row in rref:
            i = -1
            ent = 0
            if not self.allzero(row):
                # finds the leading entries (pivot positions)
                while ent == 0:
                    i += 1
                    ent = row[i]
                pivots.append(i)
        cols = self.get_self_columns()
        return [cols[i] for i in pivots]

    def getVectors(self) -> list:
        """
        Returns the set of vectors consisting of the self matrix.

        Returns
        -------
        vectors: list
            A list of vectors.
        """
        return self.get_self_columns()

    def isLinIndependent(self) -> bool:
        """
        Identifies if the self-matrix is linearly independent or not.

        Returns
        -------
        bool
            True if the self matrix is linearly independent and False if it is linearly dependent.

        """
        if self.rref_matrix is None:
            self.getRREF()
        nr, nc = self.getDimensions()
        if nr < nc:
            return False
        # scan one through each row and check if each has a pivot
        columns = get_columns(self.rref_matrix)
        for i in range(nc):
            piv = columns[i][i]
            # if the selected entry is 0 (not diagonal) or not all other entries in the same column are zero
            if piv == 0 or not all(columns[i][k] == 0 for k in range(len(columns[i])) if i != k):
                return False
        return True

    def isConsistent(self, vector_b: list) -> bool:
        """
        Identifies if the augmented matrix with the given vector b is consistent or not.

        Parameters
        ----------
        vector_b: list
            A vector that forms an augmented matrix with the self matrix.

        Returns
        -------
        bool
            True if the augmented matrix is consistent and False if incosistent.
        """
        aug_matrix = self.getRREF(addition=vector_b)
        # check each row if the rightmost entry is nonzero and the rest is all zero = incosistent
        for row in aug_matrix:
            if self.allzero(row[:-1]) and row[-1] != 0:
                return False
        return True

    def isLinComb(self, vector: list) -> bool:  # can a linear combination have infinitely many solutions?
        """
        Identifies if the vectors in the self matrix can form a linear combination for the given vector.

        Parameters
        ----------
        vector: list
            A vector to be checked whether a linear combination of the self matrix.

        Returns
        -------
        bool
            True if the vector is a linear combination of the self matrix and False if not.
        """
        return self.isConsistent(vector)


    def getDimensions(self) -> tuple:
        """
        Returns the dimensions of the self matrix, m and n, where m = number of rows and n = number of rows.

        Returns
        -------
        dimensions: tuple
            (m, n) where m = number of rows and n = number of columns.
        """
        return len(self.matrix), len(self.matrix[0])

    def isEigenvector(self, vector: list) -> bool:
        """
        Identifies if the given vector is an eigenvector corresponding to a certain factor lambda
        based on the self matrix.

        Parameters
        ----------
        vector: list
            A vector to be examined if an eigenvector of the self matrix.

        Returns
        -------
        bool
            True if the given vector is an eigenvector and False if not.
        """
        a_vector = self.getVectorMultip(vector)
        factor = a_vector[0] / vector[0]
        # if any divisions of an element in the new vector by a corresponding element in the original vector
        # are different from each other, there is no eigenvalue
        for i, j in zip(a_vector, vector):
            if i / j != factor:
                return False
        # if the common division is 0, then there is only trivial solution so return False
        if factor == 0:
            return False
        return True

    def isEigenvalue(self, constant) -> bool:
        """
        Identifies if the given value is an eigenvalue of the self matrix.

        Parameters
        ----------
        constant: numeric types
            A constant lambda to be examined if an eigenvalue of the self matrix.

        Returns
        -------
        bool
            True if the given constant lambda is an eigenvalue and False if not.
        """
        # A - (lambda * In) matrix
        lambdamatrix = matrix_add_sub(Matrix(self.matrix), matrix_scalar_multip(self.identity, constant), add=False)
        # if the RREF of the lambda matrix is identity -> if the lambda matrix is invertible
        if lambdamatrix.getDeterminant() != 0:
            return False
        return True


"""METHODS"""


def get_columns(matrix) -> list:
    """
    Returns the columns of the given matrix.

    Parameters
    ----------
    matrix
        A matrix whose columns are returned

    Returns
    -------
    columns: list
    """
    n = len(matrix[0])
    columns = []
    for i in range(n):
        columns.append([])
        for r in matrix:
            columns[i].append(r[i])
    return columns


def convert_fracs(matrix, decimal=False) -> Matrix:
    """
    Converts all the Fraction objects equivalent to an integer into the corresponding int type.
    Ex) 2/1: Fraction(2, 1) -> 2: int

    Parameters
    ----------
    matrix
        A matrix whose Fraction objects are converted into integers.
    decimal: bool, default=False
        If True, converts all the Fraction objects into a decimal: float

    Returns
    -------
    matrix: Matrix
        A matrix whose redundant Fraction elements are all converted into an integer.
    """
    for i in range(len(matrix)):  # index of rows
        for j in range(len(matrix[i])):  # index of columns
            entry = matrix[i][j]  # current entry
            # if the entry is Fraction type
            if type(entry) == Fraction:
                entry: Fraction
                # and if the fraction is equivalent to an interger, replace the Fraction entry with new one
                if entry.numerator % entry.denominator == 0:
                    # dtype = int
                    matrix[i][j] = math.floor(entry.numerator / entry.denominator)
                # if allowed to replace it with its decimal form, replace the Fraction entry with new one
                elif decimal is True:
                    # dtype = float
                    matrix[i][j] = entry.numerator / entry.denominator
    return matrix


def vect_scalar_multip(vector: list, constant):
    """
    Computes vector-scalar multiplication.

    Parameters
    ----------
    vector: list
        A vector to be scaled.
    scalar: numerical data types
        A real number to be multiplied with the vector.

    Returns
    -------
    vector: list
        Multiplied vector with the given scalar.
    """
    for i in range(len(vector)):
        vector[i] *= constant
    return vector


def vector_add_sub(vector1:list, vector2:list, add=True) -> list:
    """
    Computes the vector addition/subtraction. Vector 1 and 2 should have the same dimensions.

    Parameters
    ----------
    vector1: list
        A vector to be added to o rsubtracted from the other.
    vector2: list
        Another vector to be added to or subtracted from the other.

    Raises
    ------
    ValueError
        Raise ValueError if the given vectors do not have the same dimensions.

    Returns
    -------
    vector: list
        Computed vector.
    """
    # same algorithm with the method version
    if len(vector1) != len(vector2):
        raise ValueError("Dimensions are distinct")
    for i in range(len(vector1)):
        if add is True:
            vector1[i] += vector2[i]
        else:
            vector1[i] -= vector2[i]
    return vector1


def matrix_scalar_multip(matrix, constant) -> Matrix:
    """
    Computes the matrix-scalar multiplication.

    Parameters
    ----------
    matrix: Matrix, list
        A matrix to be scaled.
    scalar
        A real number to be multiplied with the matrix.

    Returns
    -------
    Matrix
        The scaled version of the given matrix.
    """
    return Matrix(np.array(matrix) * constant)


def matrix_vector_multip(matrix: Matrix, vector: list) -> list:
    """
    Computes vector-matrix multiplication between the given matrix and vector.

    Parameters
    ----------
    matrix: Matrix
        A matrix to be multiplied by the vector.
    vector: list
        A vector to be multiplied with the matrix.

    Raises
    ------
    ValueError
        Raises ValueError if the number of columns of the matrix and the dimension of the vector are distinct.

    Returns
    -------
    vector: list
        The product of the matrix and the vector.
    """
    # same algorithm with the method version
    # if the dimension of given vector is different from the number of columns of given matrix
    if len(matrix[0]) != len(vector):
        raise ValueError('Vector should have the same dimension as the columns of the standard matrix')
    columns = matrix.get_self_columns()
    for i in range(len(vector)):
        columns[i] = vect_scalar_multip(columns[i], vector[i])
    rows = get_columns(columns)
    for i in range(len(rows)):
        rows[i] = sum(rows[i])
    # namely multiplied vector
    return rows


def matrix_multip(matrix1, matrix2) -> Matrix:
    """
    Computes matrix multiplication. Matrix 1 is to be multiplied by matrix 2. Ex) (matrix 1) * (matrix 2)

    Parameters
    ----------
    matrix1: Matrix
        A matrix to be multiplied by matrix 2.
        The first matrix in the multiplicaiton formula where (matrix 1) * (matrix 2).
    matrix2
        Another matrix to be multiplied by matrix 1.
        The second matrix in the multiplicaiton formula where (matrix 1) * (matrix 2).

    Raises
    ------
    ValueError
        Raises ValueError if the number of columns of matrix 1 and the number of rows of matrix 2 are distinct.

    Returns
    -------
    Matrix
        The product matrix.

    """
    # same algorithm with the method version
    if len(matrix1[0]) != len(matrix2):
        raise ValueError('Matrix should be an n X k matrix when multiplied to the m X n matrix')
    mat1 = deepcopy(matrix1)
    columns = matrix2.get_self_columns()
    unsorted = []
    for j in range(len(columns)):
        unsorted.append(matrix_vector_multip(mat1, columns[j]))
    new = get_columns(unsorted)
    return Matrix(new)


def matrix_add_sub(matrix1: Matrix, matrix2: Matrix, add=True) -> Matrix:
    """
    Computes matrix addition.

    Parameters
    ----------
    matrix1: Matrix
        A matrix to be added to the other matrix.
    matrix2: Matrix
        Another matrix to be added to the matrix 1.
    add: bool, default=True
        True for computing addition and False for computing subtraction

    Returns
    -------
    Matrix
        The computed matrix.
    """
    # same algorithm with the method version
    if matrix1.getDimensions() != matrix2.getDimensions():
        raise ValueError("Dimensions are distinct")
    mat1 = deepcopy(matrix1)
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            if add is True:
                mat1[i][j] += matrix2[i][j]
            else:
                mat1[i][j] -= matrix2[i][j]
    return mat1


def transform(stand_matrix: Matrix, vector: list) -> list:
    """
    Computes the transformation of the given vector based on the given standard matrix.

    Parameters
    ----------
    stand_matrix: Matrix
        The standard matrix.
    vector: list
        The vector to be transformed.

    Raises
    ------
    ValueError
        Raises ValueError if the dimensions of the vector and the number of columns of the standard matrix are distinct.

    Returns
    -------
    vector: list
        The transformed vector based on the standard matrix.
    """
    if len(stand_matrix[0]) != len(vector):
        raise ValueError('Vector should have the same dimension as the columns of the standard matrix')
    return matrix_vector_multip(stand_matrix, vector)

