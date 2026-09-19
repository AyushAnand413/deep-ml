import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    #calc g(x) and g'(x)
    #calc h(x) and h'(x)
    #then we apply quotient rule 
    g = np.polyval(g_coeffs,x)
    h = np.polyval(h_coeffs,x)

    g_der_coeff = np.polyder(g_coeffs)
    h_der_coeff = np.polyder(h_coeffs)

    g_der = np.polyval(g_der_coeff,x)
    h_der = np.polyval(h_der_coeff,x)

    result = (g_der * h - g * h_der) / (h ** 2)
    return float(result)
