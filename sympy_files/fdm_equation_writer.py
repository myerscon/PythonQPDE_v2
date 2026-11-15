"""
fdm_equation_writer.py

Finite Difference Method equation writer classes for explicit and implicit schemes.

This module provides specialized equation writers that extend the base Equation_writer
class to handle finite difference approximations. Includes implementations for:
- Base FDM functionality (spatial derivative replacement, time derivative handling)
- Explicit finite difference schemes
- Implicit finite difference schemes with tridiagonal matrix systems

The classes convert symbolic PDEs to discretized forms suitable for numerical
time integration using finite difference spatial approximations.
"""

import sympy as sp
import numpy as np
from sympy_files import sympy_utils
from sympy_files.equation_writer import Equation_writer

class Fdm_equation_writer(Equation_writer):
    """
    Base class for Finite Difference Method equation writers.
    
    Extends Equation_writer with common finite difference functionality including
    spatial derivative replacement, time derivative handling, and conversion to
    indexed expressions suitable for numerical evaluation.
    
    Provides helper methods used by both explicit and implicit FDM implementations
    for consistent handling of spatial discretization and symbolic-to-numerical
    conversion processes.
    """
    def __init__(self, equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r=2, index_range=3, factorials=None):
        """
        Initialize finite difference equation writer.
        
        Args:
            equation_list (list): List of [variable, driver_function] pairs.
            auxiliary_list (list): List of auxiliary function definitions.
            spatial_vars_list (list): List of spatial variables.
            constants_list (list): List of constants and values.
            temporal_var: SymPy time variable.
            taylor_r (int, optional): Taylor series order. Defaults to 2.
            index_range (int, optional): Spatial stencil range. Defaults to 3.
            factorials (np.ndarray, optional): Precomputed factorials. Defaults to None.
        """
        super().__init__(equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r, index_range, factorials)

    def _replace_spatial_derivatives(self, expression, z, dz_sym, direction='central'):
        """
        Replace spatial derivatives with finite difference approximations.
        
        Systematically replaces all spatial derivatives in the expression with
        finite difference stencils for equations, auxiliaries, and spatial variables.
        
        Args:
            expression: SymPy expression containing spatial derivatives.
            z: Spatial variable symbol.
            dz_sym: Grid spacing symbol.
            direction (str, optional): Finite difference direction. Defaults to 'central'.
        
        Returns:
            SymPy expression: Expression with spatial derivatives replaced by
                finite difference approximations.
        
        Notes:
            Handles first and second derivatives for all variable types.
            Replaces derivatives in order: second derivatives first, then first
            derivatives, finally base variables.
        """
        # Handle equations
        for eqn in range(self.num_equations):
            E = self.equation_list[eqn][0]
            E_j_list = self.equation_index[self.string_equations[eqn][0]+'0']
            
            # Replace second derivatives
            expression = expression.subs(
                sp.Derivative(E,(z,2)),
                sympy_utils.second_diff_sym(E,E_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace first derivatives
            expression = expression.subs(
                sp.Derivative(E,z),
                sympy_utils.first_diff_sym(E,E_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace base variables
            expression = expression.subs(E,E_j_list[self.index_range])
            
        # Handle auxiliaries
        for aux in range(self.num_auxiliary):
            D = self.auxiliary_list[aux][0]
            D_j_list = self.auxiliary_index[self.string_auxiliary[aux][0]]
            
            # Replace second derivatives
            expression = expression.subs(
                sp.Derivative(D,(z,2)),
                sympy_utils.second_diff_sym(D,D_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace first derivatives 
            expression = expression.subs(
                sp.Derivative(D,z),
                sympy_utils.first_diff_sym(D,D_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace base variables
            expression = expression.subs(D,D_j_list[self.index_range])
            
        # Handle spatial variables
        for var in range(self.num_spatial_vars):
            S = self.spatial_vars_list[var]
            S_j_list = self.spatial_variable_index[self.string_spatial_vars[var]]
            
            # Replace second derivatives
            expression = expression.subs(
                sp.Derivative(S,(z,2)),
                sympy_utils.second_diff_sym(S,S_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace first derivatives
            expression = expression.subs(
                sp.Derivative(S,z),
                sympy_utils.first_diff_sym(S,S_j_list,dz_sym,index_range=self.index_range,direction=direction)
            )
            
            # Replace base variables
            expression = expression.subs(S,S_j_list[self.index_range])
            
        return expression

    def _replace_time_derivatives(self, expression):
        """
        Replace time derivatives with stored derivative expressions.
        
        Substitutes time derivatives of variables with their corresponding stored
        expressions from the equation index system.
        
        Args:
            expression: SymPy expression containing time derivatives.
        
        Returns:
            SymPy expression: Expression with time derivatives replaced by
                stored derivative expressions.
        
        Notes:
            Handles auxiliaries first, then equation derivatives, then spatial
            variables (assumed constant in time).
        """
        # Replace auxiliary time derivatives
        #for aux2 in range(self.num_auxiliary): unused loop - commented out for now
        for aux in range(self.num_auxiliary):
            current_aux = self.auxiliary_index[self.string_auxiliary[aux][0]]
            current_aux_deriv = self.auxiliary_index[self.string_auxiliary[aux][0]+'_deriv']
            for idx in range(2*self.index_range+1):
                expression = expression.subs(
                    sp.Derivative(current_aux[idx],self.t),
                    current_aux_deriv[idx]
                )
                    
        # Replace equation time derivatives
        for eqn2 in range(self.num_equations):
            for tylr2 in range(self.taylor_r):
                current_var = self.equation_index[self.string_equations[eqn2][0]+str(tylr2)]
                current_deriv = self.equation_index[self.string_equations[eqn2][0]+str(tylr2+1)]
                for idx in range(2*self.index_range+1):
                    expression = expression.subs(
                        sp.Derivative(current_var[idx],self.t),
                        current_deriv[idx]
                    )
                    
        # Handle spatial variables (assumed constant in time)
        for var in range(self.num_spatial_vars):
            current_var = self.spatial_variable_index[self.string_spatial_vars[var]]
            for idx in range(2*self.index_range+1):
                expression = expression.subs(
                    sp.Derivative(current_var[idx],self.t),
                    0
                )
                
        return expression

    def _create_indexed_expression(self, expression):
        """
        Convert expression to use indexed variables for numerical evaluation.
        
        Transforms symbolic expressions to use IndexedBase variables that can
        be evaluated as numerical arrays, enabling lambda function creation.
        
        Args:
            expression: SymPy expression using standard variables.
        
        Returns:
            SymPy expression: Expression using indexed variables suitable for
                lambda function generation.
        
        Notes:
            Replaces continuous variables with discretized array-based variables.
            Substitutes physical constants with their numerical values.
            Order: equations and derivatives, auxiliaries, spatial variables, constants.
        """
        var_index = self.disc_var_list['j']
        indexed_expr = expression
        
        # Index equations and derivatives
        for eqn in range(self.num_equations):
            for tylr in range(self.taylor_r+1):
                E_index = self.disc_var_list[self.string_equations[eqn][0]+str(tylr)+'_']
                E_list = self.equation_index[self.string_equations[eqn][0]+str(tylr)]
                indexed_expr = sympy_utils.index_subs(indexed_expr,E_list,E_index,var_index,self.index_range)
                
        # Index auxiliaries
        for aux in range(self.num_auxiliary):
            D_index = self.disc_var_list[self.string_auxiliary[aux][0]+'_']
            D_list = self.auxiliary_index[self.string_auxiliary[aux][0]]
            indexed_expr = sympy_utils.index_subs(indexed_expr,D_list,D_index,var_index,self.index_range)
            
        # Index spatial variables
        for var in range(self.num_spatial_vars):
            S_index = self.disc_var_list[self.string_spatial_vars[var]+'_']
            S_list = self.spatial_variable_index[self.string_spatial_vars[var]]
            indexed_expr = sympy_utils.index_subs(indexed_expr,S_list,S_index,var_index,self.index_range)
            
        # Replace constants
        for constant in range(self.num_constants):
            indexed_expr = indexed_expr.subs(
                self.constants_list[0][constant],
                self.constants_list[1][constant]
            )
            
        return indexed_expr
    
    def _create_lambdified_expression(self, indexed_expr):
        """
        Create lambdified function from indexed expression.
        
        Generates a lambda function that can be called with numerical arrays.
        All possible variables are included as arguments for consistency across
        all lambda functions in the system.
        
        Args:
            indexed_expr: SymPy expression using indexed variables.
        
        Returns:
            callable: Lambda function that accepts numerical arrays and returns
                computed values.
        
        Notes:
            Argument order: index j, spatial variables, auxiliaries, equations
            and derivatives. Consistent ordering ensures all lambda functions
            can be called with the same argument structure.
        """
        # Create argument list for lambdification
        lambdify_args = [self.disc_var_list['j']]  # Start with index j
        
        # Add ALL spatial variables first
        for var in range(self.num_spatial_vars):
            lambdify_args.append(self.disc_var_list[self.string_spatial_vars[var]+'_'])
        
        # Add ALL auxiliary variables
        for aux in range(self.num_auxiliary):
            lambdify_args.append(self.disc_var_list[self.string_auxiliary[aux][0]+'_'])
        
        # Add ALL equations and their derivatives
        for eqn in range(self.num_equations):
            for tylr in range(self.taylor_r+1):
                lambdify_args.append(self.disc_var_list[self.string_equations[eqn][0]+str(tylr)+'_'])
        
        return sp.lambdify(lambdify_args, indexed_expr, modules=['numpy'])

class Explicit_fdm_equation_writer(Fdm_equation_writer):
    """
    Explicit finite difference method equation writer.
    
    Implements explicit time stepping schemes where derivatives are computed
    directly from known values at the current time step. Creates lambda
    functions for evaluating driver functions and their time derivatives.
    """
    def __init__(self, equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r=2, index_range=3, factorials=None):
        super().__init__(equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r, index_range, factorials)

    def finite_diff_approx(self, z, dz_sym, direction='central'):
        """
        Create explicit finite difference approximations for all equations.
        
        Processes all equations and auxiliary functions to create explicit finite
        difference approximations. Generates lambda functions for driver functions
        and their time derivatives up to order taylor_r.
        
        Args:
            z: Spatial variable symbol.
            dz_sym: Grid spacing symbol.
            direction (str, optional): Finite difference direction. Defaults to 'central'.
        
        Returns:
            None
        
        Side Effects:
            Creates self.d_dict with auxiliary function lambda functions.
            Creates self.f_dict with equation and derivative lambda functions.
        
        Notes:
            For explicit schemes, time derivatives are computed by differentiating
            the driver function expressions and substituting known time derivatives.
            All lambda functions use consistent argument ordering for evaluation.
        """
        d_dict = {}
        f_dict = {}
        
        # Process auxiliary functions
        for aux in range(self.num_auxiliary):
            f_D = self.auxiliary_list[aux][1]
            d_dict['d_'+str(aux)] = f_D
            
            # Replace derivatives and variables
            f_D = self._replace_spatial_derivatives(f_D, z, dz_sym, direction)
            f_D_indexed = self._create_indexed_expression(f_D)
            
            d_dict['d_'+str(aux)+'_indexed'] = f_D_indexed
            d_dict['d_'+str(aux)+'_lambdified'] = self._create_lambdified_expression(f_D_indexed)
            
        setattr(self, 'd_dict', d_dict)
        
        # Process equations and their derivatives
        for eqn in range(self.num_equations):
            f = self.equation_list[eqn][1]
            
            # Replace spatial derivatives
            f = self._replace_spatial_derivatives(f, z, dz_sym, direction)
            f_dict['f'+str(eqn)+'_0'] = f
            
            # Create indexed version
            f_indexed = self._create_indexed_expression(f)
            f_dict['f'+str(eqn)+'_0_indexed'] = f_indexed
            f_dict['f'+str(eqn)+'_0_lambdified'] = self._create_lambdified_expression(f_indexed)
            
            # Handle time derivatives
            current_f = f
            for tylr in range(self.taylor_r):
                # Take time derivative
                current_f = current_f.diff(self.t)
                
                # Replace time derivatives with stored expressions
                current_f = self._replace_time_derivatives(current_f)
                f_dict['f'+str(eqn)+'_'+str(tylr+1)] = current_f
                
                # Create indexed version
                current_f_indexed = self._create_indexed_expression(current_f)
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_indexed'] = current_f_indexed
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_lambdified'] = self._create_lambdified_expression(current_f_indexed)
                
        setattr(self, 'f_dict', f_dict)

class Implicit_fdm_equation_writer(Fdm_equation_writer):
    """
    Implicit finite difference method equation writer.
    
    Implements implicit time stepping schemes that require solving tridiagonal
    matrix systems. Creates coefficients for implicit matrix equations and
    integrand functions for quantum integration of correction terms.
    """
    def __init__(self, equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r=2, index_range=3, factorials=None):
        super().__init__(equation_list, auxiliary_list, spatial_vars_list, constants_list, temporal_var, taylor_r, index_range, factorials)

    def _create_sympy_alpha_expression(self, t, t0, C):
        """
        Create symbolic expression for alpha using Taylor series expansion.
        
        Args:
            t: Time variable.
            t0: Initial time.
            C: Coefficient array.
        
        Returns:
            SymPy expression: Taylor series for alpha function.
        """
        t_diff = t - t0
        alpha_expr = 0
        for i in range(self.taylor_r + 2):
            alpha_expr += C[i] * (t_diff ** i) / self.factorials[i]
        return alpha_expr

    def _create_sympy_alpha_dot_expression(self, t, t0, C):
        """
        Create symbolic expression for alpha time derivative.
        
        Args:
            t: Time variable.
            t0: Initial time.
            C: Coefficient array.
        
        Returns:
            SymPy expression: Time derivative of alpha Taylor series.
        """
        t_diff = t - t0
        alpha_dot_expr = 0
        for i in range(1, self.taylor_r + 2):
            alpha_dot_expr += C[i] * i * (t_diff ** (i-1)) / self.factorials[i]
        return alpha_dot_expr
    
    def _create_alpha_expressions_variables(self):
        """
        Create symbolic variables for alpha and alpha_dot expressions.
        
        Sets up time variable t0 and coefficient symbols C for different
        grid points (j-1, j, j+1) used in implicit scheme expressions.
        
        Returns:
            None
        
        Side Effects:
            Creates self.t0 and self.C_symbols for implicit integration.
        """
        self.t0 = sp.Symbol('t0')
        # One independent set of Taylor-coefficient symbols per equation, for j-1, j, j+1
        n = self.taylor_r + 2
        self.C_symbols = [
            [[sp.Symbol(f'C{e}_{lab},{i}') for i in range(n)] for lab in ('j-1', 'j', 'j+1')]
            for e in range(self.num_equations)
        ]
        # Boundary alpha / alpha_dot placeholders, one per equation
        self.alpha_boundary_symbols = {
            name: [sp.Symbol(f'{name}_{e}') for e in range(self.num_equations)]
            for name in ('alpha_left', 'alpha_right', 'alpha_dot_left', 'alpha_dot_right')
        }

    def _create_alpha_expressions_list(self):
        """
        Create lists of alpha and alpha_dot expressions for grid points.
        
        Generates symbolic expressions for alpha and its time derivative
        at different spatial locations (j-1, j, j+1) for implicit schemes.
        
        Returns:
            None
        
        Side Effects:
            Creates self.alpha_expr_list and self.alpha_dot_expr_list.
        """
        self.alpha_expr_list = []
        self.alpha_dot_expr_list = []
        for e in range(self.num_equations):
            Cs = self.C_symbols[e]
            self.alpha_expr_list.append([self._create_sympy_alpha_expression(self.t, self.t0, C) for C in Cs])
            self.alpha_dot_expr_list.append([self._create_sympy_alpha_dot_expression(self.t, self.t0, C) for C in Cs])

    def _substitute_alpha_expressions_list(self, expr):
        """
        Substitute each equation's own alpha / alpha_dot Taylor polynomials into an indexed expression.

        Every variable (E0_{j-1}, T0_{j}, ..., E1_{j+1}, ...) is replaced by the polynomial built
        from that equation's coefficient symbols C{eqn}_*. At the left (right) boundary the j-1 (j+1)
        values are replaced by the per-equation boundary placeholders alpha_left_{eqn}
        (alpha_right_{eqn}) and their derivatives by alpha_dot_*_{eqn}.

        Returns:
            list: [left_expr, middle_expr, right_expr].
        """
        j = sp.Symbol('j')
        offsets = [sp.Add(j, sp.Integer(-1)), j, sp.Add(j, sp.Integer(1))]
        bsym = self.alpha_boundary_symbols

        def build(left_boundary, right_boundary):
            out = expr.copy()
            for deriv, exprs, b_l, b_r in ((0, self.alpha_expr_list, 'alpha_left', 'alpha_right'),
                                           (1, self.alpha_dot_expr_list, 'alpha_dot_left', 'alpha_dot_right')):
                for e in range(self.num_equations):
                    base = sp.IndexedBase(sp.Symbol(self.string_equations[e][0] + str(deriv)))
                    vals = list(exprs[e])
                    if left_boundary:
                        vals[0] = bsym[b_l][e]
                    if right_boundary:
                        vals[2] = bsym[b_r][e]
                    for off, val in zip(offsets, vals):
                        out = out.subs(sp.Indexed(base, off), val)
            return out

        return [build(True, False), build(False, False), build(False, True)]

    def _t_plus_hbar_expansion(self, expression, expand_aux=False):
        """
        Process expression by expanding variables to t+hbar using Taylor series.
        
        Handles each term separately to preserve structure while expanding variables
        from time t to t+hbar for implicit time stepping schemes.
        
        Args:
            expression: SymPy expression to process.
            expand_aux (bool, optional): Whether to expand auxiliary variables. Defaults to False.
        
        Returns:
            SymPy expression: Expression with variables expanded to t+hbar.
        
        Notes:
            For equations: u(t+h) = u(t) + h*u'(t)
            Processes terms individually to avoid structural issues with fractions.
            Only expands numerators when denominators are present.
        """
        # Declare hbar as a Sympy symbol
        hbar = sp.Symbol(r"\bar{h}")
        
        # Build substitution dictionary for all variables
        var_subs = {}
        for eqn in range(self.num_equations):
            base_name = self.string_equations[eqn][0]
            var_list = self.equation_index[base_name+'0']
            deriv_list = self.equation_index[base_name+'1']
            
            for idx in range(2*self.index_range + 1):
                var = var_list[idx]
                expanded_var = var + hbar*deriv_list[idx]
                var_subs[var] = expanded_var
        
        # Add auxiliaries to substitution dictionary if needed
        if expand_aux:
            for aux in range(self.num_auxiliary):
                base_name = self.string_auxiliary[aux][0]
                aux_list = self.auxiliary_index[base_name]
                
                for idx in range(2*self.index_range + 1):
                    aux_var = aux_list[idx]
                    expanded_aux = aux_var + hbar*aux_var.diff(self.t)
                    var_subs[aux_var] = expanded_aux
        
        # Process each term in the expression separately
        if expression.is_Add:
            result = 0
            for term in expression.args:
                result += self._process_term(term, var_subs)
            return result
        else:
            # Single term case
            return self._process_term(expression, var_subs)
        
    def _process_term(self, term, var_subs):
        """
        Process a single term in expression, safely handling symbolic expressions.
        
        Separates numerator and denominator to avoid expanding variables in
        denominators, which can cause numerical issues in implicit schemes.
        
        Args:
            term: Single term from the expression.
            var_subs (dict): Dictionary mapping variables to their expanded forms.
        
        Returns:
            SymPy expression: Processed term with appropriate expansions.
        
        Notes:
            If no denominator exists, expands everything.
            If denominator present, only expands numerator to maintain stability.
        """
        # Try to separate numerator and denominator safely
        num, den = term.as_numer_denom()
        
        # If denominator is just 1, the term has no fraction
        if den == 1:
            # No denominator - expand everything
            result = term
            for var, expanded in var_subs.items():
                result = result.subs(var, expanded)
            return result
        else:
            # There's a denominator - only expand in the numerator
            expanded_num = num
            for var, expanded in var_subs.items():
                expanded_num = expanded_num.subs(var, expanded)
            return expanded_num / den
    
    def replace_infinities(self, expr):
        """
        Replace problematic infinity values with zero.
        
        Handles numerical issues that can arise in symbolic processing by
        replacing complex infinities and problematic power expressions.
        
        Args:
            expr: SymPy expression to clean.
        
        Returns:
            SymPy expression: Expression with infinities replaced by zero.
        
        Notes:
            Replaces complex infinity (zoo) and 0^x expressions.
            Essential for numerical stability in implicit coefficient extraction.
        """
        expr = expr.subs(sp.zoo, 0)
        #expr = expr.subs(sp.oo, 0)
        #expr = expr.subs(sp.nan, 0)

        # Also replace any 0**x with 0 (assuming x is positive or a symbol) 4/30
        expr = expr.replace(lambda e: isinstance(e, sp.Pow) and e.args[0] == 0 and 
                                (e.args[1].is_positive or e.args[1].is_Symbol), 
                            lambda e: 0)

        return expr
    
    def _extract_implicit_coefficients_and_Kj(self, func, deriv_list, var_list):
        """
        Extract coefficients for tridiagonal system from implicit equation.
        
        Decomposes the implicit equation into coefficients for the tridiagonal
        matrix system and the remaining terms (Kj) for the right-hand side.
        
        Args:
            func: Implicit equation expression.
            deriv_list: List of derivative terms at j-1, j, j+1.
            var_list: List of all variables in the system.
        
        Returns:
            list: [coeff_list, Kj] where:
                - coeff_list: [a_, b_, c_] coefficients for tridiagonal matrix
                - Kj: Remaining terms after derivative coefficients removed
        
        Notes:
            For j-th equation: a_*u_{j-1} + b_*u_j + c_*u_{j+1} = Kj
            Handles complex infinities that may arise from derivative terms.
        """
        # Initialize coefficient list for the tridiagonal system
        coeff_list = []
        deriv_list_length = 3  # j-1, j, j+1 terms
        #var_list_length = self.num_equations * 3
        f = func

        # Extract coefficients in front of derivative terms
        for i in range(deriv_list_length):
            if i==1:
                this_expr = 1-f.coeff(deriv_list[i])
                coeff_list.append(this_expr)
            else:
                this_expr = -f.coeff(deriv_list[i])
                coeff_list.append(this_expr)

        # Remaining terms after derivative terms are removed compose Kj:
        for i in range(deriv_list_length):
            f = f.subs(deriv_list[i],0)
            # Remove complex infinities in case of denominator deriv terms:
            f = self.replace_infinities(f)

        return [coeff_list, f]
    
    def _create_primed_coeffs(self,coeff_list, eqn):
        """
        Create primed coefficient expressions for implicit integration.
        
        Takes tridiagonal coefficients [a_, b_, c_] and creates their time
        derivatives with proper substitutions and auxiliary replacements.
        
        Args:
            coeff_list (list): [a_, b_, c_] coefficient expressions.
            eqn (int): Equation index for proper variable substitution.
        
        Returns:
            list: [a_prime, b_prime, c_prime] time derivatives of coefficients.
        
        Notes:
            Each primed coefficient includes the corresponding E_{j-1}, E_j, or E_{j+1}
            factor to complete the integrand for implicit correction terms.
            Handles auxiliary function substitutions and time derivative replacements.
        """
        pass
        [a_, b_, c_] = coeff_list
        
        # Substitute auxiliaries for their expressions
        a_ = self._replace_indexed_auxiliaries(a_)
        b_ = self._replace_indexed_auxiliaries(b_)
        c_ = self._replace_indexed_auxiliaries(c_)

        # Take Sympy derivatives
        a_prime = a_.diff(self.t)
        b_prime = b_.diff(self.t)
        c_prime = c_.diff(self.t)

        # Replaced discretized variables with their indexed versions
        a_prime = self._replace_time_derivatives_indexed(a_prime)
        b_prime = self._replace_time_derivatives_indexed(b_prime)
        c_prime = self._replace_time_derivatives_indexed(c_prime)


        # We must add E_{j-1}, E_{j}, and E_{j+1} to a_prime, b_prime, and c_prime, respectively (see implicit derivation notes - pending publication)
        additional_term_list = self.equation_index[self.string_equations[eqn][0]+"0"]
        # each primed term needs a corresponding factor of E to complete the integrand. See notes.
        a_prime = a_prime.expand() * additional_term_list[0]
        b_prime = b_prime.expand() * additional_term_list[1]
        c_prime = c_prime.expand() * additional_term_list[2]

        return [a_prime, b_prime, c_prime]

    def _replace_indexed_auxiliaries(self, expression):
        """
        Replace indexed auxiliary functions with their expressions.
        
        This function is needed for handling time derivatives of auxiliaries in the implicit scheme,
        particularly for the 'primed coefficients' of the tridiagonal system. Substitutes auxiliary 
        functions like D_{j-1}, D_j, D_{j+1} with their corresponding expressions.
        
        Args:
            expression: SymPy expression containing indexed auxiliary functions.
        
        Returns:
            SymPy expression: Expression with auxiliaries replaced by their definitions.
        
        Notes:
            Handles time derivatives of auxiliaries in implicit schemes.
            Creates shifted versions of auxiliary expressions for different grid points.
            Essential for primed coefficient computation where auxiliaries appear.
        """
        # Create a mapping of auxiliary function patterns and their expressions
        aux_subs = {}
        
        # Loop through all auxiliary functions
        for aux in range(self.num_auxiliary):
            # Get the auxiliary function name and expression
            aux_name = self.string_auxiliary[aux][0]
            aux_expr = self.auxiliary_list[aux][1]
            
            # Create patterns for j-1, j, and j+1 versions of the auxiliary
            for offset in [-1, 0, 1]:
                if offset == -1:
                    symbol = sp.Function(f"{aux_name}_{{j-1}}")(self.t)
                elif offset == 0:
                    symbol = sp.Function(f"{aux_name}_{{j}}")(self.t)
                else:  # offset == 1
                    symbol = sp.Function(f"{aux_name}_{{j+1}}")(self.t)
                
                # Create a modified expression with appropriate variable shifts
                shifted_expr = aux_expr
                
                # Replace variables in the expression with shifted versions
                for eqn in range(self.num_equations):
                    E = self.equation_list[eqn][0]
                    E_shifted = sp.Function(f"{E.name}_{{j{offset:+d}}}")(self.t) if offset != 0 else sp.Function(f"{E.name}_{{j}}")(self.t)
                    shifted_expr = shifted_expr.subs(E, E_shifted)
                
                # Also handle shifts for spatial variables
                for var in range(self.num_spatial_vars):
                    S = self.spatial_vars_list[var]
                    S_shifted = sp.Function(f"{S.name}_{{j{offset:+d}}}")(self.t) if offset != 0 else sp.Function(f"{S.name}_{{j}}")(self.t)
                    shifted_expr = shifted_expr.subs(S, S_shifted)
                
                # Add to substitution dictionary
                aux_subs[symbol] = shifted_expr
        
        # Apply all substitutions
        result = expression
        for symbol, replacement in aux_subs.items():
            result = result.subs(symbol, replacement)
        
        return result
    
    def _replace_time_derivatives_indexed(self, expression):
        """
        Replace indexed time derivatives with numbered variable representations.
        
        Converts time derivatives of indexed variables to their numbered forms
        for use in implicit scheme coefficient expressions.
        
        Args:
            expression: SymPy expression containing indexed derivatives.
        
        Returns:
            SymPy expression: Expression with derivatives in numbered form.
        
        Notes:
            Maps: E_{j}(t) -> E0_{j}, Derivative(E_{j}(t), t) -> E1_{j}, etc.
            Processes in order: second derivatives, first derivatives, base variables.
            Handles both regular and already-numbered derivative patterns.
        """
        result = expression
        
        # Process each equation
        for eqn in range(self.num_equations):
            # Substitute for each variable in each equation
            base_name = self.string_equations[eqn][0]
            
            # Loop through possible j-offset patterns
            for j_notation in ["j-1", "j", "j+1"]:                
                # Replace second derivative
                second_deriv = sp.Derivative(sp.Function(f"{base_name}_{{{j_notation}}}")(self.t), (self.t, 2))
                result = result.subs(second_deriv, sp.Function(f"{base_name}2_{{{j_notation}}}")(self.t))
                
                # Replace first derivative
                first_deriv = sp.Derivative(sp.Function(f"{base_name}_{{{j_notation}}}")(self.t), self.t)
                result = result.subs(first_deriv, sp.Function(f"{base_name}1_{{{j_notation}}}")(self.t))

                # Additional replace first derivative step - sometimes we end up with sp.Derivative(T0_{j},t) instead of just sp.Derivative(T_{j},t)
                first_deriv_new = sp.Derivative(sp.Function(f"{base_name}0_{{{j_notation}}}")(self.t), self.t)
                result = result.subs(first_deriv_new, sp.Function(f"{base_name}1_{{{j_notation}}}")(self.t))
                
                # Replace the base function last
                var_pattern = sp.Function(f"{base_name}_{{{j_notation}}}")(self.t)
                result = result.subs(var_pattern, sp.Function(f"{base_name}0_{{{j_notation}}}")(self.t))
        
        return result

    def _create_integrand_lambdified_expression(self, expr):
        """
        Create lambda function for integrand expressions with boundary values.
        
        Generates lambda functions for primed coefficient integrands that include
        time, initial time, coefficients, and boundary condition parameters.
        
        Args:
            expr: SymPy integrand expression.
        
        Returns:
            callable: Lambda function with signature:
                (t, t0, *coeffs_all_equations, alpha_left, alpha_right, alpha_dot_left, alpha_dot_right)
        
        Notes:
            Used for quantum integration of primed coefficient expressions in
            implicit schemes. Includes boundary condition handling for alpha values.
        """
        flat_C = [c for eq in self.C_symbols for pos in eq for c in pos]
        bsym = self.alpha_boundary_symbols
        boundary = (bsym['alpha_left'] + bsym['alpha_right'] +
                    bsym['alpha_dot_left'] + bsym['alpha_dot_right'])
        base = sp.lambdify([self.t, self.t0] + flat_C + boundary, expr, modules=['numpy'])
        n_coeffs = len(flat_C)

        def integrand(t, t0, *args):
            # args = all-equation coefficients, then alpha_left, alpha_right, alpha_dot_left,
            # alpha_dot_right, each a scalar or a length-num_equations array
            bounds = [b for arr in args[n_coeffs:n_coeffs + 4] for b in np.broadcast_to(np.asarray(arr, dtype=float), (self.num_equations,))]
            return base(t, t0, *args[:n_coeffs], *bounds)

        return integrand

    def implicit_finite_diff_approx(self, z, dz_sym, direction='central'):
        """
        Create implicit finite difference approximations with tridiagonal systems.
        
        Main method for implicit scheme setup. Processes equations to create
        tridiagonal matrix coefficients, primed integrands, and lambda functions
        needed for implicit time stepping with quantum integration.
        
        Args:
            z: Spatial variable.
            dz_sym: Grid spacing symbol.
            direction (str, optional): Finite difference direction. Defaults to 'central'.
        
        Returns:
            None
        
        Side Effects:
            Creates self.d_dict with auxiliary functions.
            Creates self.c_dict with tridiagonal coefficients and primed integrands.
            Creates self.f_dict with main equations and derivatives.
        
        Notes:
            For each equation, the process:
            1. Replaces spatial derivatives with finite differences
            2. Applies t+hbar expansion for implicit scheme
            3. Extracts tridiagonal coefficients [a_, b_, c_]
            4. Creates primed coefficient expressions for integration
            5. Generates lambda functions for all components
            
            Handles multi-equation coupling by substituting derivative terms
            between equations during the coefficient extraction process.
        """
        # Time-step symbol used by _t_plus_hbar_expansion (must be the same symbol)
        hbar = sp.Symbol(r"\bar{h}")

        # Initialize dictionaries for implicit forms
        d_dict = {}
        c_dict = {}
        f_dict = {}

        self._create_alpha_expressions_variables()
        self._create_alpha_expressions_list()
                
        # Process auxiliary functions
        for aux in range(self.num_auxiliary):
            f_D = self.auxiliary_list[aux][1]
            d_dict['d_'+str(aux)] = f_D
                        
            # Replace derivatives and variables
            f_D = self._replace_spatial_derivatives(f_D, z, dz_sym, direction)
            f_D_indexed = self._create_indexed_expression(f_D)
            d_dict['d_'+str(aux)+'_indexed'] = f_D_indexed
            d_dict['d_'+str(aux)+'_lambdified'] = self._create_lambdified_expression(f_D_indexed)
        
        setattr(self, 'd_dict', d_dict)

        # Pre-pass: spatial-derivative form of each driver function ('f<eqn>_break'),
        # needed when substituting other equations' time derivatives below
        for eqn in range(self.num_equations):
            f = self.equation_list[eqn][1]
            f_dict['f'+str(eqn)+'_break'] = self._replace_spatial_derivatives(f, z, dz_sym, direction)

        # Process equations
        for eqn in range(self.num_equations):
            f = self.equation_list[eqn][1]

            deriv_list = self.equation_index[self.string_equations[eqn][0]+'1']
            eqn_list = []
            for eqn2 in range(self.num_equations):
                sub_eqn_list = self.equation_index[self.string_equations[eqn2][0]+'0']
                for item in sub_eqn_list:
                    eqn_list.append(item)
            
            # Step 1: Replace spatial derivatives 
            f = self._replace_spatial_derivatives(f, z, dz_sym, direction)
            
            # Step 2: Apply time plus hbar expansion
            f = self._t_plus_hbar_expansion(f)
            
            # Step 3: Expand symbolically
            f = f.expand()
            
            # Step 4: Set hbar**2 terms to zero
            f = f.subs(hbar**2, 0)

            # Step 4a: Replace off variable derivative terms with f expression at current time t
            for eqn2 in range(self.num_equations):
                if (eqn2 != eqn):
                    f = f.subs(self.equation_index[self.string_equations[eqn2][0]+'1'][1],f_dict['f'+str(eqn2)+'_break'])
            
            # Step 5: Extract coefficients and K_j
            [[a_, b_, c_], f] = self._extract_implicit_coefficients_and_Kj(f, deriv_list, eqn_list)

            # Create primed versions of the coefficients
            [a_prime,b_prime,c_prime] = self._create_primed_coeffs([a_, b_, c_], eqn)
            
            f_dict['f'+str(eqn)+'_0'] = f
            c_dict['a_'+str(eqn)] = a_
            c_dict['b_'+str(eqn)] = b_
            c_dict['c_'+str(eqn)] = c_
            c_dict['a_'+str(eqn)+'_prime'] = a_prime
            c_dict['b_'+str(eqn)+'_prime'] = b_prime
            c_dict['c_'+str(eqn)+'_prime'] = c_prime

            # Create indexed versions
            f_indexed = self._create_indexed_expression(f)
            a_indexed = self._create_indexed_expression(a_)
            b_indexed = self._create_indexed_expression(b_)
            c_indexed = self._create_indexed_expression(c_)
            a_prime_indexed = self._create_indexed_expression(a_prime)
            b_prime_indexed = self._create_indexed_expression(b_prime)
            c_prime_indexed = self._create_indexed_expression(c_prime)

            # Store indexed versions
            f_dict['f'+str(eqn)+'_0_indexed'] = f_indexed
            c_dict['a_'+str(eqn)+'_indexed'] = a_indexed
            c_dict['b_'+str(eqn)+'_indexed'] = b_indexed
            c_dict['c_'+str(eqn)+'_indexed'] = c_indexed
            c_dict['a_'+str(eqn)+'_prime_integrand_indexed'] = a_prime_indexed
            c_dict['b_'+str(eqn)+'_prime_integrand_indexed'] = b_prime_indexed
            c_dict['c_'+str(eqn)+'_prime_integrand_indexed'] = c_prime_indexed

            [a_prime_integrand_left,a_prime_integrand_middle,a_prime_integrand_right] = self._substitute_alpha_expressions_list(a_prime_indexed)
            [b_prime_integrand_left,b_prime_integrand_middle,b_prime_integrand_right] = self._substitute_alpha_expressions_list(b_prime_indexed)
            [c_prime_integrand_left,c_prime_integrand_middle,c_prime_integrand_right] = self._substitute_alpha_expressions_list(c_prime_indexed)

            c_dict['a_'+str(eqn)+'_prime_integrand_expr_left'] = a_prime_integrand_left
            c_dict['a_'+str(eqn)+'_prime_integrand_expr_middle'] = a_prime_integrand_middle
            c_dict['a_'+str(eqn)+'_prime_integrand_expr_right'] = a_prime_integrand_right

            c_dict['b_'+str(eqn)+'_prime_integrand_expr_left'] = b_prime_integrand_left
            c_dict['b_'+str(eqn)+'_prime_integrand_expr_middle'] = b_prime_integrand_middle
            c_dict['b_'+str(eqn)+'_prime_integrand_expr_right'] = b_prime_integrand_right

            c_dict['c_'+str(eqn)+'_prime_integrand_expr_left'] = c_prime_integrand_left
            c_dict['c_'+str(eqn)+'_prime_integrand_expr_middle'] = c_prime_integrand_middle
            c_dict['c_'+str(eqn)+'_prime_integrand_expr_right'] = c_prime_integrand_right

            # Create base lambda functions
            f_lambda = self._create_lambdified_expression(f_indexed)
            a_lambda = self._create_lambdified_expression(a_indexed)
            b_lambda = self._create_lambdified_expression(b_indexed)
            c_lambda = self._create_lambdified_expression(c_indexed)

            # Store wrapped lambda functions that ensure array output
            f_dict['f'+str(eqn)+'_0_lambdified'] = f_lambda
            c_dict['a_'+str(eqn)+'_lambdified'] = lambda *args: np.full(len(args[0]), a_lambda(*args))
            c_dict['b_'+str(eqn)+'_lambdified'] = lambda *args: np.full(len(args[0]), b_lambda(*args))
            c_dict['c_'+str(eqn)+'_lambdified'] = lambda *args: np.full(len(args[0]), c_lambda(*args))
            
            # Create lambda functions for integrand expressions
            for pos in ['left', 'middle', 'right']:
                for coeff in ['a', 'b', 'c']:
                    expr_key = f'{coeff}_'+str(eqn)+f'_prime_integrand_expr_{pos}'
                    lambda_key = f'{coeff}_'+str(eqn)+f'_prime_integrand_lambda_{pos}'
                    c_dict[lambda_key] = self._create_integrand_lambdified_expression(c_dict[expr_key])
            
            # Handle time derivatives
            current_f = f
            for tylr in range(self.taylor_r):
                # Take time derivative
                current_f = current_f.diff(self.t)
                
                # Replace time derivatives with stored expressions
                current_f = self._replace_time_derivatives(current_f)
                
                f_dict['f'+str(eqn)+'_'+str(tylr+1)] = current_f
                
                # Create indexed version
                current_f_indexed = self._create_indexed_expression(current_f)
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_indexed'] = current_f_indexed
                f_dict['f'+str(eqn)+'_'+str(tylr+1)+'_lambdified'] = self._create_lambdified_expression(current_f_indexed)
        
        setattr(self, 'c_dict', c_dict)
        setattr(self, 'f_dict', f_dict)
