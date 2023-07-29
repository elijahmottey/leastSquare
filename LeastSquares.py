import numpy as np

#Index Number:PS/CSC/19/0076
#Assignment 3


def solve_normal_equation(A, b):
    """
    Return the solution x to the linear least squares problem
    Ax ≈ b using normal equations,
    where A is an (m x n) matrix, with m > n, rank(A) = n, and
    b is a vector of size (m)
    """
    # Step 1: Calculate the normal equation (A^T * A) * x = A^T * b
    A_transpose = A.T
    ATA = np.dot(A_transpose, A)
    ATb = np.dot(A_transpose, b)

    # Step 2: Perform LU decomposition
    L, U = lu_decomposition(ATA)

    # Step 3: Solve the triangular system L * y = ATb
    y = solve_triangular(L, ATb, lowerTriangularMatrix=True)

    # Step 4: Solve the triangular system U * x = y
    x = solve_triangular(U, y, lowerTriangularMatrix=False)

    return x

def lu_decomposition(A):
    """
    Perform LU decomposition on the matrix A.
    Returns L and U matrices such that A = L * U.
    """
    # Implement LU decomposition (Crout's algorithm)
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

def solve_triangular(A, b, lowerTriangularMatrix=True):
    """
    Solve the system of linear equations Ax = b, where A is a triangular matrix.
    where A = [L][U]
    If lowerTrangularMatrix=True, A is a lower triangular matrix; otherwise, it's an upper triangular matrix.
    """
    n = A.shape[0]
    x = np.zeros(n)

    if lowerTriangularMatrix:
        for i in range(n):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i))) / A[i, i]
    else:
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i + 1, n))) / A[i, i]

    return x

# Given data

irisA = np.array([[4.8, 6.3, 5.7, 5.1, 7.7, 5.6, 4.9, 4.4, 6.4, 6.2, 6.7, 4.5, 6.3,
                  4.8, 5.8, 4.7, 5.4, 7.4, 6.4, 6.3, 5.1, 5.7, 6.5, 5.5, 7.2, 6.9,
                  6.8, 6. , 5. , 5.4, 5.6, 6.1, 5.9, 5.6, 6. , 6. , 4.4, 6.9, 6.7,
                  5.1, 6. , 6.3, 4.6, 6.7, 5. , 6.7, 5.8, 5.1, 5.2, 6.1],
                 [3.1, 2.5, 3. , 3.7, 3.8, 3. , 3.1, 2.9, 2.9, 2.9, 3.1, 2.3, 2.5,
                  3.4, 2.7, 3.2, 3.7, 2.8, 2.8, 3.3, 2.5, 4.4, 3. , 2.6, 3.2, 3.2,
                  3. , 2.9, 3.6, 3.9, 3. , 2.6, 3.2, 2.8, 2.7, 2.2, 3.2, 3.1, 3.1,
                  3.8, 2.2, 2.7, 3.1, 3. , 3.5, 2.5, 4. , 3.5, 3.4, 3. ],
                 [1.6, 4.9, 4.2, 1.5, 6.7, 4.5, 1.5, 1.4, 4.3, 4.3, 5.6, 1.3, 5. ,
                  1.6, 5.1, 1.3, 1.5, 6.1, 5.6, 6. , 3. , 1.5, 5.8, 4.4, 6. , 5.7,
                  5.5, 4.5, 1.4, 1.3, 4.1, 5.6, 4.8, 4.9, 5.1, 5. , 1.3, 4.9, 4.4,
                  1.5, 4. , 4.9, 1.5, 5.2, 1.3, 5.8, 1.2, 1.4, 1.4, 4.9]])

irisb = np.array([0.2, 1.5, 1.2, 0.4, 2.2, 1.5, 0.1, 0.2, 1.3, 1.3, 2.4, 0.3, 1.9,
                    0.2, 1.9, 0.2, 0.2, 1.9, 2.1, 2.5, 1.1, 0.4, 2.2, 1.2, 1.8, 2.3,
                    2.1, 1.5, 0.2, 0.4, 1.3, 1.4, 1.8, 2. , 1.6, 1.5, 0.2, 1.5, 1.4,
                    0.3, 1. , 1.8, 0.2, 2.3, 0.3, 1.8, 0.2, 0.2, 0.2, 1.8])

# Call the function to solve the least squares problem
iris_x = solve_normal_equation(irisA.T, irisb)
iris_residual = np.linalg.norm(np.dot(irisA.T, iris_x) - irisb)

print("\nSolution vector iris_x:\n")
print(iris_x)

print("\n2-norm of the residual iris_residual:")
print(iris_residual)
