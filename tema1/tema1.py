import pathlib
import math

def parse_equation(equation: str) -> tuple[list[float], float]:
    equation = equation.replace(' ', '')

    coefficients = [0.0, 0.0, 0.0]
    constant = 0.0

    current_num = ''
    sign = 1

    for char in equation:
        if char.isdigit() or char == '.':
            current_num += char
        elif char in ['x', 'y', 'z']:
            if current_num == '':
                current_num = '1'
            coefficients[['x', 'y', 'z'].index(char)] = sign * float(current_num)
            current_num = ''
            sign = 1
        elif char == '+':
            sign = 1
        elif char == '-':
            sign = -1
        elif char == '=':
            current_num = equation[equation.index(char) + 1:]
            constant = sign * float(current_num)
            break

    return coefficients, constant

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    A = []
    B = []

    with open(path, 'r') as file:
        for line in file:
            coefficients, constant = parse_equation(line.strip())
            A.append(coefficients)
            B.append(constant)

    return A, B

A, B = load_system(pathlib.Path("system.txt"));
print(f"A={A}")
print(f"B={B}")


def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) != 3 or len(matrix[0]) != 3:
        raise ValueError("The matrix must be 3x3.")
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]

    det = (a11 * (a22 * a33 - a23 * a32)
           - a12 * (a21 * a33 - a23 * a31)
           + a13 * (a21 * a32 - a22 * a31))
    return det

det_A = determinant(A)
print(f"determinant(A) = {det_A}")


def trace(matrix: list[list[float]]) -> float:
    if len(matrix) != 3 or len(matrix[0]) != 3:
        raise ValueError("The matrix must be 3x3.")
    trace_value = matrix[0][0] + matrix[1][1] + matrix[2][2]
    return trace_value

trace_A = trace(A)
print(f"trace(A) = {trace_A}")


def norm(vector: list[float]) -> float:
    sum_of_squares = sum(b ** 2 for b in vector)
    return math.sqrt(sum_of_squares)

norm_B = norm(B)
print(f"norm(B) = {norm_B}")


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if len(matrix) != 3 or len(matrix[0]) != 3:
        raise ValueError("The matrix must be 3x3.")
    transposed = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            transposed[j][i] = matrix[i][j]
    return transposed

transposed_A = transpose(A)
print(f"transpose(A) = {transposed_A}")


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if len(matrix) != 3 or len(matrix[0]) != 3 or len(vector) != 3:
        raise ValueError("Matrix must be 3x3 and vector must have 3 elements.")
    result = [0 for _ in range(3)]
    for i in range(3):
        result[i] = sum(matrix[i][j] * vector[j] for j in range(3))
    return result

result = multiply(A, B)
print(f"multiply(A, B) = {result}")


def replace_column(matrix: list[list[float]], column: int, vector: list[float]) -> list[list[float]]:
    new_matrix = [row[:] for row in matrix]
    for i in range(len(matrix)):
        new_matrix[i][column] = vector[i]
    return new_matrix


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    det_A = determinant(matrix)

    if det_A == 0:
        raise ValueError("det(A) = 0 => system has multiple solutions.")

    A_x = replace_column(matrix, 0, vector)
    A_y = replace_column(matrix, 1, vector)
    A_z = replace_column(matrix, 2, vector)

    det_A_x = determinant(A_x)
    det_A_y = determinant(A_y)
    det_A_z = determinant(A_z)

    x = det_A_x / det_A
    y = det_A_y / det_A
    z = det_A_z / det_A

    return [x, y, z]

solution = solve_cramer(A, B)
print(f"solve_cramer(A, B) = {solution}")

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

def determinant_2x2(matrix: list[list[float]]) -> float:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    cofactor_matrix = []
    for i in range(3):
        row = []
        for j in range(3):
            minor_matrix = minor(matrix, i, j)
            cofactor_value = ((-1) ** (i + j)) * determinant_2x2(minor_matrix)
            row.append(cofactor_value)
        cofactor_matrix.append(row)
    return cofactor_matrix

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    cof_matrix = cofactor(matrix)
    adj_matrix = [[cof_matrix[j][i] for j in range(3)] for i in range(3)]
    return adj_matrix

def inverse(matrix: list[list[float]]) -> list[list[float]]:
    det_A = determinant(matrix)
    if det_A == 0:
        raise ValueError("det(A) = 0 => A not invertible.")

    adj_matrix = adjoint(matrix)
    inv_matrix = [[adj_matrix[i][j] / det_A for j in range(3)] for i in range(3)]
    return inv_matrix

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inv_matrix = inverse(matrix)
    solution = multiply(inv_matrix, vector)
    return solution

solution = solve(A, B)
print(f"solve_inversion(A, B) = {solution}")

