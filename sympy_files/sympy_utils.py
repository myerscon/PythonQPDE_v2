"""
sympy_utils.py

Custom SymPy functions for handling finite difference approximations and derivative substitutions.

This module provides utilities for:
- Converting continuous derivatives to finite difference approximations
- Managing indexed symbolic variables for spatial discretization
- Symbolic manipulation functions for equation processing

Used by the equation writer classes to transform continuous PDEs into discrete
finite difference equations suitable for numerical solution.
"""


### Finite Difference Method utilities

def first_diff_sym(func, func_list, delta, index_range, direction='central'):
    """
    Create first-order finite difference approximation with discrete variable renaming.
    
    Converts continuous first derivatives to finite difference expressions using
    specified stencil direction and grid spacing.
    
    Args:
        func: Name of SymPy function being differentiated.
        func_list (list): List of discretized SymPy functions at stencil points:
            [f_{j-3}, f_{j-2}, f_{j-1}, f_{j}, f_{j+1}, f_{j+2}, f_{j+3}]
        delta: Grid spacing symbol (dx, dr, etc.).
        index_range (int): Index range for accessing func_list elements.
        direction (str, optional): Finite difference type. Defaults to 'central'.
            Options: 'central', 'backward', 'forward', 'backward_second_order', 
                    'forward_second_order'
    
    Returns:
        SymPy expression: Finite difference approximation of first derivative.
    
    Notes:
        Central difference: (f_{j+1} - f_{j-1}) / (2*delta)
        Backward: (f_j - f_{j-1}) / delta
        Forward: (f_{j+1} - f_j) / delta
        Second-order variants use 3-point stencils for improved accuracy.
    """
    j = index_range
    if (direction == 'central'):
        return ((func_list[j+1] - func_list[j-1]) / (2*delta))
    elif (direction == 'backward'):
        return ((func_list[j] - func_list[j-1]) / (delta))
    elif (direction == 'forward'):
        return ((func_list[j+1] - func_list[j-1]) / (delta))
    elif (direction == 'backward_second_order'):
        return ((3*func_list[j] - 4*func_list[j-1] + func_list[j-2]) / (2*delta))
    elif (direction == 'forward_second_order'):
        return ((-3*func_list[j] + 4*func_list[j+1] - func_list[j+1]) / (2*delta))
    else:
        print("Error: not a valid direction variable")

def second_diff_sym(func, func_list, delta, index_range, direction='central'):
    """
    Create second-order finite difference approximation with discrete variable renaming.
    
    Converts continuous second derivatives to finite difference expressions using
    specified stencil direction and grid spacing.
    
    Args:
        func: Name of SymPy function being differentiated.
        func_list (list): List of discretized SymPy functions at stencil points:
            [f_{j-3}, f_{j-2}, f_{j-1}, f_{j}, f_{j+1}, f_{j+2}, f_{j+3}]
        delta: Grid spacing symbol.
        index_range (int): Index range for accessing func_list elements.
        direction (str, optional): Finite difference type. Defaults to 'central'.
            Options: 'central', 'backward_second_order', 'forward_second_order'
    
    Returns:
        SymPy expression: Finite difference approximation of second derivative.
    
    Notes:
        Central difference: (f_{j+1} - 2*f_j + f_{j-1}) / delta^2
        Second-order backward/forward use 4-point stencils for boundary treatment.
    """
    j = index_range
    if (direction == 'central'):
        return ((func_list[j+1] - 2*func_list[j] + func_list[j-1]) / (delta**2))
    elif (direction == 'backward_second_order'):
        return ((-2*func_list[j] + 5*func_list[j-1] - 4*func_list[j-2] + func_list[j-3]) / (delta**2))
    elif (direction == 'forward_second_order'):
        return ((2*func_list[j] - 5*func_list[j+1] + 4*func_list[j+2] - func_list[j+3]) / (delta**2))
    else:
        print("Error: not a valid direction variable")

### General sympy utilities

def index_subs(func, var_list, var_index, index, index_range):
    """
    Replace variables in expression with their indexed versions at specified index.
    
    Substitutes each element in the variable list with its corresponding indexed
    variable, enabling conversion from continuous to discrete representations.
    
    Args:
        func: SymPy function/expression for substitution.
        var_list (list): List of variables to be replaced.
        var_index: Indexed function for substituting into func.
        index: Index variable for var_index.
        index_range (int): Range of indices used in finite difference stencil.
            For index_range=3: indices span (j-3, ..., j+3)
    
    Returns:
        SymPy expression: Function with variables replaced by indexed versions.
    
    Notes:
        Maps continuous variables to discrete grid points for finite difference
        approximations. Essential for converting symbolic PDEs to numerical form.
    """
    return_func = func
    for i in range(len(var_list)):
        return_func = return_func.subs(var_list[i],var_index[index+(i-index_range)])
    return return_func

# Currently unused:

def list_subs(func, var_1_list, var_2_list):
    """
    Replace elements of var_1_list in expression with corresponding elements from var_2_list.
    
    Performs element-wise substitution between two variable lists of equal length.
    
    Args:
        func: SymPy function/expression for substitution.
        var_1_list (list): List of variables to be replaced.
        var_2_list (list): List of replacement variables.
    
    Returns:
        SymPy expression: Function with substitutions applied.
    
    Notes:
        Currently unused in the main codebase but available for custom
        symbolic manipulations.
    """
    return_func = func
    for i in range(len(var_1_list)):
        return_func = return_func.subs(var_1_list[i],var_2_list[i])
    return return_func

def list_collect(func, var_list):
    """
    Apply SymPy 'collect' operation for each variable in the list.
    
    Groups terms by specified variables to simplify expressions and reveal
    coefficient structure.
    
    Args:
        func: SymPy function/expression for collection.
        var_list (list): List of variables to collect terms by.
    
    Returns:
        SymPy expression: Function with terms collected by each variable.
    
    Notes:
        Useful for organizing expressions by powers of variables or 
        extracting coefficients in finite difference equations.
        Currently unused but available for symbolic processing.
    """
    return_func = func
    for i in range(len(var_list)):
        return_func = return_func.collect(var_list[i])
    return return_func

def list_subs_func(func,expr1,expr2,expr_list):
    """
    Replace expr1 with expr2 in func for each element in expr_list.
    
    Performs repeated substitutions of expression patterns, useful for
    systematic symbolic transformations.
    
    Args:
        func: SymPy function for substitution.
        expr1: Expression pattern to be replaced.
        expr2: Replacement expression.
        expr_list (list): List of elements to process (currently unused in implementation).
    
    Returns:
        SymPy expression: Function with substitutions applied.
    
    Notes:
        Implementation appears incomplete - expr_list parameter is not used.
        Currently unused in the main codebase.
    """
    return_func = func
    for i in range(len(expr_list)):
        return_func = return_func.subs(expr1,expr2)
        pass