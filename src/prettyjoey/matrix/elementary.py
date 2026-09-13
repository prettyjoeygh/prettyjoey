import torch
# matrix as input, index of source row, and target row
def rowswap(matrix, source_row_index, target_index):
    # fixed bug, I think memory overwrites destroys good data
    # using [[]] fixes this overwriting issue
    matrix[[source_row_index, target_index]] = matrix[[target_index, source_row_index]]
    return matrix 



# input matrix, index of source row, and scaling factor
def rowscale(matrix, source_row_index, scaling_factor):
    matrix[source_row_index] =  matrix[source_row_index] * scaling_factor
    return matrix

def rowreplacement(matrix, row_i, row_j, j,k):
    """
    perform jRi + kRj that takes a matrix as an
    input, first row, the second row, the scaling factors j and k. You may use function
    rowscale for scaling operation.
    """
    # cannot perform both using rowscale in two instances because the scaling factor will
    # permanently alter row_j in the matrix, when row_j should remain unchanged
    # only row_i is supposed to be replaced by the operation jRi + kRj
    rowscale(matrix,row_i,j) # Ri -> j * Ri
    matrix[row_i] = matrix[row_i] + k * matrix[row_j] # ... + k*Rj
    return matrix

def rref(matrix):
    """
    notes:
    ref:
        1. all 0 rows at bottom
        2. pviot of all non 0 rows are on the left of all rows below
        3. pivot must be a value of "1"
    rref:
        everything above + 
        4. each pivot column has zero's everywhere else
    
    """
    # python struggles to do the math without a locked type as float
    # was throwing garbage values out without it
    matrix = matrix.to(torch.float64)
    # recall: shape returns tuple of values returns: (rows,cols)
    row_len, col_len = matrix.shape
    locked_rows = []
    # 1: iterate over column starting in the top left
    for col_i in range(col_len):
        # 2a: select pivot row
        pivot_row_i = -1  # -1 means "no pivot found yet" for this column

        # search for a row (not already used as a pivot) with a nonzero entry here
        for row_i in range(row_len):
            # skip this row, will -> 0 regardless
            if row_i in locked_rows:
                continue
            # check if current element is a non zero value
            if (matrix[row_i][col_i]) != 0:
                pivot_row_i = row_i
                break

        # no usable pivot in this column, go to the next 
        if pivot_row_i == -1:
            continue

        # 2b: bring pivot row -> locked_row indices
        next_pivot_row_i = (locked_rows[-1] + 1) if len(locked_rows) > 0 else 0

        # if not already there, swap it into place with rowswap funct.
        # if we find a non zero pivot in a subsequent row we need to verfiy the previous row has its pivot in place, if no we need to move it up
        if pivot_row_i != next_pivot_row_i:
            rowswap(matrix, next_pivot_row_i, pivot_row_i)
            pivot_row_i = next_pivot_row_i  # pivot moved, update index

        # 3: scale the pivot row so its leading entry becomes 1 via. rowscale funct.
        pivot_val = matrix[pivot_row_i][col_i]
        # eqn: 
        rowscale(matrix, pivot_row_i, 1 / pivot_val)

        # 4: eliminate this column in every other rows with rowreplacement funct.
        for row_i in range(row_len):
            if row_i == pivot_row_i:
                continue
            lead_coefficient = matrix[row_i][col_i]
            if (lead_coefficient) != 0:
                rowreplacement(matrix, row_i, pivot_row_i, 1, -lead_coefficient)
        # 5: prep to move onto next column
        locked_rows.append(pivot_row_i)

    
    return matrix


def main():
    # testing 
    """
    test = torch.tensor([ [1, 3, 0, 0, 3],[0, 0, 1, 0, 9],[0, 0, 0, 1, -4] ], dtype=torch.float64)
    print(test)
    test2 = rowswap(test,0,1) #correctly swaps rows 1 & 2
    print("\n", test2)
    test3 = rowscale(test2, 0, 1 / 3)
    print("\n",test3)
    test4 = rowreplacement(test3, 2, 0, 1, -3)
    print("\n",test4)
if __name__ == "__main__":
    main()
    
    """
