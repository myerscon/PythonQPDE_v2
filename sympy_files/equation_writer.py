"""
equation_writer.py

Base equation writer class for creating discretized approximations from symbolic equations.

This module provides the foundational Equation_writer class that processes symbolic
equations, auxiliary functions, and spatial variables to create discretized
representations suitable for finite difference methods.

The class handles:
- Parsing symbolic equation definitions from input files
- Creating indexed variable representations for spatial discretization
- Managing auxiliary functions and their derivatives
- Setting up the framework for finite difference approximations
"""

import sympy as sp

class Equation_writer():
    """
    Base class for converting symbolic equations to discretized finite difference form.
    
    Processes symbolic equation definitions, auxiliary functions, spatial variables,
    and constants to create indexed representations suitable for numerical solution.
    Creates the foundation for finite difference approximations by organizing
    variables into discretized arrays spanning spatial stencil points.
    
    Attributes:
        taylor_r (int): Highest derivative order in Taylor series expansion.
        index_range (int): Spatial stencil range (±index_range grid points).
        t: Temporal variable symbol.
        equation_list: Input list of [variable, driver_function] pairs.
        num_equations (int): Number of equations in the system.
        auxiliary_list: Input list of auxiliary function definitions.
        num_auxiliary (int): Number of auxiliary functions.
        spatial_vars_list: Input list of spatial variables.
        num_spatial_vars (int): Number of spatial variables.
        constants_list: Input list of constants and their values.
        num_constants (int): Number of constants.
        equation_index (dict): Indexed equation variables for discretization.
        auxiliary_index (dict): Indexed auxiliary variables and derivatives.
        spatial_variable_index (dict): Indexed spatial variables.
        disc_var_list (dict): Discretized variable symbols for final equations.
    """
    def __init__(self,equation_list,auxiliary_list,spatial_vars_list,constants_list,temporal_var,taylor_r=2,index_range=3,factorials=None) -> None:
        """
        Initialize equation writer with symbolic definitions and parameters.
        
        Sets up the equation writer by processing input symbolic definitions and
        creating indexed representations for all variables, equations, and auxiliary
        functions needed for finite difference approximations.
        
        Args:
            equation_list (list): List of [variable, driver_function] pairs for each equation.
            auxiliary_list (list): List of [variable, expression, derivative] for auxiliaries.
            spatial_vars_list (list): List of spatial variable symbols.
            constants_list (list): List of [symbols, values] for physical constants.
            temporal_var: SymPy symbol for time variable.
            taylor_r (int, optional): Highest Taylor series derivative order. Defaults to 2.
            index_range (int, optional): Spatial stencil half-width. Defaults to 3.
            factorials (np.ndarray, optional): Precomputed factorial values. Defaults to None.
        
        Returns:
            None
        
        Notes:
            Creates indexed representations spanning (j-index_range) to (j+index_range)
            for all variables, enabling finite difference stencil operations.
        """
        self.taylor_r = taylor_r
        self.index_range = index_range
        self.t = temporal_var
        self.factorials = factorials  # Store the factorials array 3/24
        self.equation_list = equation_list
        [self.string_equations,self.num_equations] = self._create_equation_list(equation_list)
        self.auxiliary_list = auxiliary_list
        [self.string_auxiliary,self.num_auxiliary] = self._create_auxiliary_list(auxiliary_list)
        self.spatial_vars_list = spatial_vars_list
        [self.string_spatial_vars,self.num_spatial_vars] = self._create_spatial_variable_list(spatial_vars_list)
        self.constants_list = constants_list
        [self.string_constants,self.num_constants] = self._create_constants_list(constants_list)
        self.equation_index = self._create_equation_index_list(temporal_var,self.string_equations,self.num_equations,taylor_r,index_range)
        self.auxiliary_index = self._create_auxiliary_index_list(temporal_var,self.string_auxiliary,self.num_auxiliary,auxiliary_list,self.string_equations,self.num_equations,equation_list,self.equation_index,index_range)
        self.spatial_variable_index = self._create_spatial_variable_index_list(self.string_spatial_vars,self.num_spatial_vars,index_range)
        self.disc_var_list = self._create_discretized_var_list(self.string_equations,self.num_equations,self.string_auxiliary,self.num_auxiliary,self.string_spatial_vars,self.num_spatial_vars,taylor_r)

    def _create_equation_list(self,equation_list):
        """
        Extract equation variable names from equation definitions.
        
        Processes the input equation list to extract just the variable names,
        excluding the driver function expressions.
        
        Args:
            equation_list (list): List of [variable, driver_function] pairs.
        
        Returns:
            list: [string_equations, num_equations] where:
                - string_equations (list): Variable names as strings
                - num_equations (int): Number of equations
        
        Notes:
            Only extracts variable names (e.g., 'T', 'E') not the full expressions.
            Driver functions are processed separately during finite difference setup.
        """
        string_equations = []
        num_equations = len(equation_list)
        for eqn in range(num_equations):
            string_list = []
            string_list.append(equation_list[eqn][0].name) # no need for name creation for f, only E
            string_equations.append(string_list)

        return [string_equations,num_equations]

    def _create_auxiliary_list(self,auxiliary_list):
        """
        Extract auxiliary function names from auxiliary definitions.
        
        Processes the input auxiliary list to extract variable names,
        excluding expressions and derivatives.
        
        Args:
            auxiliary_list (list): List of [variable, expression, derivative] for auxiliaries.
        
        Returns:
            list: [string_auxiliary, num_auxiliary] where:
                - string_auxiliary (list): Auxiliary variable names as strings
                - num_auxiliary (int): Number of auxiliary functions
        
        Notes:
            Extracts only the auxiliary variable names for indexing purposes.
            Expressions and derivatives are processed during equation setup.
        """
        string_auxiliary = []
        num_auxiliary = len(auxiliary_list)
        for aux in range(num_auxiliary):
            string_list = []
            string_list.append(auxiliary_list[aux][0].name) # do not need names for auxiliary derivatives
            string_auxiliary.append(string_list)

        return [string_auxiliary,num_auxiliary]

    def _create_spatial_variable_list(self,spatial_vars_list):
        """
        Extract spatial variable names and count from input list.
        
        Args:
            spatial_vars_list (list): List of SymPy spatial variable symbols.
        
        Returns:
            list: [string_spatial_vars, num_spatial_vars] where:
                - string_spatial_vars (list): Spatial variable names as strings
                - num_spatial_vars (int): Number of spatial variables
        
        Notes:
            Prepares spatial variables for multi-dimensional finite difference
            extensions and grid indexing operations.
        """
        string_spatial_vars = []
        num_spatial_vars = len(spatial_vars_list)
        for var in range(num_spatial_vars):
            string_spatial_vars.append(spatial_vars_list[var].name)

        return [string_spatial_vars,num_spatial_vars]

    def _create_constants_list(self,constants_list):
        """
        Extract constant names and count from input definitions.
        
        Args:
            constants_list (list): List containing [symbols, values] for constants.
        
        Returns:
            list: [string_constants, num_constants] where:
                - string_constants (list): Constant names as strings
                - num_constants (int): Number of constants
        
        Notes:
            Constants include physical parameters, grid spacing, and other
            simulation parameters used in the equations.
        """
        string_constants = []
        num_constants = len(constants_list[0])
        for const in range(num_constants):
            string_constants.append(constants_list[0][const].name)

        return [string_constants,num_constants]

    def _create_equation_index_list(self,t,string_equations,num_equations,taylor_r,index_range):
        """
        Create indexed equation variables for finite difference stencils.
        
        Generates arrays of SymPy functions for each equation variable and their
        time derivatives, discretized across spatial stencil points. Essential
        for finite difference approximation of spatial derivatives.
        
        Args:
            t: SymPy time variable symbol.
            string_equations (list): Equation variable names.
            num_equations (int): Number of equations.
            taylor_r (int): Maximum Taylor series derivative order.
            index_range (int): Spatial stencil half-width.
        
        Returns:
            dict: equation_index dictionary with keys like 'f0', 'f1', 'f2' for
                base variables and derivatives, each containing arrays of indexed
                functions spanning spatial stencil points.
        
        Notes:
            Creates functions like T0_{j-1}, T0_{j}, T0_{j+1} for base variables
            and T1_{j-1}, T1_{j}, T1_{j+1} for first derivatives, etc.
            Used by finite difference approximation functions.
        """
        equation_index = {}
        for eqn in range(num_equations):
            for tylr in range(taylor_r+1):
                next_list = []
                next_list_name = string_equations[eqn][0] + str(tylr)
                for idx in range(2*(index_range)+1):
                    if (idx<(index_range)):
                        next_list.append(sp.Function(next_list_name+'_{j-'+str(index_range-idx)+'}')(self.t))
                    elif (idx>(index_range)):
                        next_list.append(sp.Function(next_list_name+'_{j+'+str(idx-index_range)+'}')(self.t))
                    else:
                        next_list.append(sp.Function(next_list_name+'_{j}')(self.t))
                equation_index[next_list_name] = next_list

        return equation_index
    
    def _create_auxiliary_index_list(self,t,string_auxiliary,num_auxiliary,auxiliary_list,string_equations,num_equations,equation_list,equation_index,index_range):
        """
        Create indexed auxiliary variables and their derivatives.
        
        Similar to equation indexing but also processes auxiliary derivative
        expressions by substituting indexed equation variables. This enables
        proper handling of auxiliary functions that depend on main variables.
        
        Args:
            t: SymPy time variable symbol.
            string_auxiliary (list): Auxiliary variable names.
            num_auxiliary (int): Number of auxiliary functions.
            auxiliary_list (list): Full auxiliary definitions with expressions.
            string_equations (list): Equation variable names.
            num_equations (int): Number of equations.
            equation_list (list): Full equation definitions.
            equation_index (dict): Previously created equation indices.
            index_range (int): Spatial stencil half-width.
        
        Returns:
            dict: auxiliary_index dictionary containing both auxiliary variables
                and their derivatives ('_deriv' suffix) with proper substitutions.
        
        Notes:
            Handles coupling between auxiliary functions and main equations by
            substituting indexed variables into auxiliary derivative expressions.
        """
        # Creates an indexed list of auxiliary functions from an input function.
        auxiliary_index = {}
        for aux in range(num_auxiliary):
            # create auxiliary index dictionary
            next_list = []
            next_list_name = string_auxiliary[aux][0]
            for idx in range(2*(index_range)+1):
                if (idx<(index_range)):
                    next_list.append(sp.Function(next_list_name+'_{j-'+str(index_range-idx)+'}')(self.t)) # j-1,j-2,...
                elif (idx>(index_range)):
                    next_list.append(sp.Function(next_list_name+'_{j+'+str(idx-index_range)+'}')(self.t)) # j+1,j+2,...
                else:
                    next_list.append(sp.Function(next_list_name+'_{j}')(self.t)) # j
            auxiliary_index[next_list_name] = next_list
        # create auxiliary derivative index dictionary
        #     now that we have dictionaries containing all variables and auxiliary variables, we can create a list of expressions containing
        #     the derivatives of auxiliary variables at each index point
        for aux in range(num_auxiliary):
            deriv_list = []
            deriv_list_name = string_auxiliary[aux][0] + '_deriv'
            deriv = auxiliary_list[aux][2]
            for idx in range(2*(index_range)+1):
                deriv_k = deriv
                for eqn in range(num_equations):
                    deriv_k = deriv_k.subs(equation_list[eqn][0],equation_index[string_equations[eqn][0]+'0'][idx])
                for aux in range(num_auxiliary):
                    deriv_k = deriv_k.subs(auxiliary_list[aux][0],auxiliary_index[string_auxiliary[aux][0]][idx])
                deriv_list.append(deriv_k)
            auxiliary_index[deriv_list_name] = deriv_list

        return auxiliary_index
    
    def _create_spatial_variable_index_list(self,string_spatial_vars,num_spatial_vars,index_range):
        """
        Create indexed spatial variables for grid discretization.
        
        Generates indexed versions of spatial variables (x, r, etc.) across
        stencil points for use in finite difference operations.
        
        Args:
            string_spatial_vars (list): Spatial variable names.
            num_spatial_vars (int): Number of spatial variables.
            index_range (int): Spatial stencil half-width.
        
        Returns:
            dict: spatial_variable_index dictionary with indexed spatial variables.
        
        Notes:
            Important for multi-dimensional extensions where spatial coordinates
            themselves may vary across grid points (e.g., cylindrical coordinates).
        """
        spatial_variable_index = {}
        for var in range(num_spatial_vars):
            next_list = []
            next_list_name = string_spatial_vars[var]
            for idx in range(2*(index_range)+1):
                if (idx<(index_range)):
                     next_list.append(sp.Function(next_list_name+'_{j-'+str(index_range-idx)+'}')(self.t)) # j-1,j-2,...
                elif (idx>(index_range)):
                    next_list.append(sp.Function(next_list_name+'_{j+'+str(idx-index_range)+'}')(self.t)) # j+1,j+2,...
                else:
                    next_list.append(sp.Function(next_list_name+'_{j}')(self.t)) # j
            spatial_variable_index[next_list_name] = next_list

        return spatial_variable_index
    
    def _create_discretized_var_list(self,string_equations,num_equations,string_auxiliary,num_auxiliary,string_spatial_vars,num_spatial_vars,taylor_r):
        """
        Create discretized SymPy variables for lambda function generation.
        
        Creates discretized SymPy variables that allow for the creation of lambda
        functions which can be called in Python to calculate the functions. The
        discretized variables are an essential step in converting the continuous
        in time, discontinuous in space SymPy functions into functions that are
        discretized in both space and time and can be used to advance the simulation
        forward in time to find the solution.
        
        Args:
            string_equations (list): Equation variable names.
            num_equations (int): Number of equations.
            string_auxiliary (list): Auxiliary variable names.
            num_auxiliary (int): Number of auxiliary functions.
            string_spatial_vars (list): Spatial variable names.
            num_spatial_vars (int): Number of spatial variables.
            taylor_r (int): Maximum Taylor derivative order.
        
        Returns:
            dict: disc_var_list dictionary containing discretized variable symbols
                with '_' suffix (e.g., T0_, T1_, D_) and index variable 'j'.
        
        Notes:
            The discretized SymPy variables are named by adding '_' to the end of
            each variable name. An index variable 'j' is also created for use in
            the discretized form of the equations.
        """
        disc_var_list = {}
        disc_var_list['j'] = sp.Symbol("j") # index variable
        for eqn in range(num_equations):
            for tylr in range(taylor_r+1):
                disc_var_list[string_equations[eqn][0]+str(tylr)+'_'] = sp.IndexedBase(string_equations[eqn][0]+str(tylr))
        for aux in range(num_auxiliary):
            disc_var_list[string_auxiliary[aux][0]+'_'] = sp.IndexedBase(string_auxiliary[aux][0])
        for var in range(num_spatial_vars):
            disc_var_list[string_spatial_vars[var]+'_'] = sp.IndexedBase(string_spatial_vars[var])

        return disc_var_list
    