import linearalgebra as la

matrix = la.Matrix([[1, 1, 0, 0, 0, 0],
                   [4, 1, 1, 0, 0, 0],
                   [8, 0, 1, 1, 0, 0],
                   [8, -2, 0, -1, 1, 1],
                   [4, 0, -2, 0, -1, 0]])

print(f'matrix = \n{matrix}')
print('matrix.getRREF()\n>>>')
print(matrix.getRREF())