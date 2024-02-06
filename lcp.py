# Practice linear regression model, cost function and gradient descent
import math, copy
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Features - 1000s of square feet
    x_train = np.array([1.0, 2.0])
    # Targets - 1000s of dollars
    y_train = np.array([300.0, 500.0])

    w_init=  0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # Run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# Cost function
def compute_cost(x, y, w, b):
    # Number of samples
    m = x.shape[0]
    cost = 0

    # Compute cost for each sample
    for i in range(m):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    
    return (1/(2*m) * cost)

# Compute Gradient
def compute_gradient(x,y,w,b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
     
    # Number of samples
    m = x.shape[0]
    dj_dw=0
    dj_db=0
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i])*x[i]
        dj_db_i = (f_wb - y[i])
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db
# Utilize gradient and compute cost
def gradient_descent(x,y,w_in,b_in,alpha,num_iters, cost_function,gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history=[]
    b= b_in
    w=w_in
    
    for i in range(num_iters):
        #Compute Gradient
        dj_dw, dj_db = gradient_function(x,y,w,b)
        
        b= b-alpha*dj_db
        w= w-alpha*dj_dw
        
        # Save cost J at each iteration
        if i<num_iters-1:
            J_history.append(cost_function(x,y,w,b))
        else:
            J_history.append(cost_function(x,y,w_in,b_in))
            p_history.append([w,b])
            
        # Print the cost every 10 or as many iterations if <10
        if i %math.ceil(num_iters/10) ==0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:10.8f}",
                  f"dj_dw {dj_dw:0.3e}, dj_db {dj_db:0.3e}",
                  f"w {w:0.3e}, b {b:0.3e}")
    return w, b, J_history, p_history #return w and J,w history for graphing

if __name__ == "__main__":
    main()