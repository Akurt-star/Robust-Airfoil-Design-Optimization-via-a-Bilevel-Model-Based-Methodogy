import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import csv
from scipy import linalg as LA

#FGN : The problem expressed here in form   min f(x) : c(x)>=0

# Define the problem parameters
n = 5
mu = np.array([
    8.6033358901938017e-01, 3.4256184594817283e+00, 6.4372981791719468e+00,
    9.5293344053619631e+00, 1.2645287223856643e+01, 1.5771284874815882e+01,
    1.8902409956860023e+01, 2.2036496727938566e+01, 2.5172446326646664e+01,
    2.8309642854452012e+01, 3.1447714637546234e+01, 3.4586424215288922e+01,
    3.7725612827776501e+01, 4.0865170330488070e+01, 4.4005017920830845e+01,
    4.7145097736761031e+01, 5.0285366337773652e+01, 5.3425790477394663e+01,
    5.6566344279821521e+01, 5.9707007305335459e+01, 6.2847763194454451e+01,
    6.5988598698490392e+01, 6.9129502973895256e+01, 7.2270467060308960e+01,
    7.5411483488848148e+01, 7.8552545984242926e+01, 8.1693649235601683e+01,
    8.4834788718042290e+01, 8.7975960552493220e+01, 9.1117161394464745e+01
])

def objective(x):
    """Calculate the objective function value."""
    return np.sum(x**2)

def rho(x, mu):
    """Calculate the rho function for given x and mu values."""
    n = len(x)
    result = np.zeros(30)
    for j in range(30):
        exp_term = np.exp(-mu[j]**2 * np.sum(x**2))
        sum_term = sum(2 * (-1)**(ii-1) * np.exp(-mu[j]**2 * np.sum(x[ii-1:]**2)) for ii in range(2, n+1))
        result[j] = -(exp_term + sum_term + (-1)**n) / mu[j]**2
    return result

def A(mu_val):
    """Calculate the A function for a given mu value."""
    return 2 * np.sin(mu_val) / (mu_val + np.sin(mu_val) * np.cos(mu_val))

def constraint(x, mu, A_values):
    """Calculate the constraint function value."""
    rho_values = rho(x, mu)

    # First term: double sum
    term1 = 0
    for i in range(30):
        for j in range(i+1, 30):
            term1 += (mu[i]**2 * mu[j]**2 * A_values[i] * A_values[j] *
                     rho_values[i] * rho_values[j] *
                     (np.sin(mu[i]+mu[j])/(mu[i]+mu[j]) +
                      np.sin(mu[i]-mu[j])/(mu[i]-mu[j]))
                    )

    # Second term: single sum
    term2 = sum(mu[j]**4 * A_values[j]**2 * rho_values[j]**2 *
                (np.sin(2*mu[j])/(2*mu[j]) + 1)/2
                for j in range(30))

    # Third term: single sum
    term3 = sum(mu[j]**2 * A_values[j] * rho_values[j] *
                (2*np.sin(mu[j])/mu[j]**3 - 2*np.cos(mu[j])/mu[j]**2)
                for j in range(30))

    return -(term1 + term2 - term3 + 2/15 - 0.0001)

def generate_data(x, radius, mu, A_values, n_samples):
    """Generate data points around current point x within given radius."""
    X = []
    y_obj = []
    y_const = []
    for _ in range(n_samples):
        delta = np.random.uniform(-radius, radius, n)
        x_new = x + delta
        X.append(x_new)
        y_obj.append(objective(x_new))
        y_const.append(constraint(x_new, mu, A_values))
    return np.array(X), np.array(y_obj), np.array(y_const)

def check_feasibility(x, mu, A_values):
    """Check if the point x satisfies the constraints."""
    constraint_value = constraint(x, mu, A_values)
    return (constraint_value <= 0)  # Check if the constraint is satisfied (g(x) <= 0)

def project_onto_feasible_region(x, mu, A_values, radius=0.1):
    """Project the point x onto the feasible region if it violates the constraints."""
    violation = constraint(x, mu, A_values)
    if violation > 0:  # If there's a violation
        # Move the point in the direction that improves feasibility
        correction = radius * violation  # Adjust correction factor as needed
        x_new = x - correction  # Adjust the position to decrease the violation
        return x_new
    return x  # If feasible, return as is

def trust_region_optimization(x0, mu, A_values, max_iterations=69, initial_radius=1.0, initial_sample = 30, new_generation= 1):
    """Main trust region optimization function with weight decay calibration."""
    # Check if initial point is feasible and project if necessary
    if not check_feasibility(x0, mu, A_values):
        print("Initial point is infeasible. Projecting onto the feasible region.")
        x0 = project_onto_feasible_region(x0, mu, A_values)

    x = x0
    radius = initial_radius
    results = []
    penalty_factor = 1000  # Penalty for constraint violations
    X_cumulative = []
    y_obj_cumulative = []
    y_const_cumulative = []

    # Lists to store values for output
    constraint_values = []
    predicted_constraint_values = []
    objective_values = []
    predicted_objective_values = []

    # Generate initial dataset
    X_init, y_obj_init, y_const_init = generate_data(x, radius, mu, A_values, n_samples = initial_sample)
    X_cumulative.extend(X_init)
    y_obj_cumulative.extend(y_obj_init)
    y_const_cumulative.extend(y_const_init)
    count = 0
    
    #FGN : Evaluate functions at the initial point and store
    current_obj = objective(x0)
    current_cons = constraint(x0, mu, A_values)
    X_cumulative.append(x0)
    y_obj_cumulative.append(current_obj)
    y_const_cumulative.append(current_cons)

    for iteration in range(max_iterations):
        # Generate one new data point
        for _ in range(new_generation):
          delta = np.random.uniform(-radius, radius, n)
          x_new = x + delta

          X_cumulative.append(x_new)
          y_obj_cumulative.append(objective(x_new))
          y_const_cumulative.append(constraint(x_new, mu, A_values))

        # Calculate adaptive weights with calibrated decay
        distances = np.linalg.norm(np.array(X_cumulative) - x, axis=1)
        alpha = 0.25  # Decay rate parameter, adjust based on problem
        weights = np.where(distances >= 2*radius, 0, np.exp(-alpha * (distances)**2))
        #weights = weights*(1/np.linalg.norm(weights,ord=np.inf))
        #weights = np.ones(np.shape(weights))
        # Create a dictionary pairing distances with weights
        distance_weight_dict = {f"Data point {i+1}": {"Distance": distances[i], "Weight": weights[i]} 
                            for i in range(len(distances))}

        # Log distances and weights
        print(f"Iteration {iteration}:")
        #for key, value in distance_weight_dict.items(): 
        #  print(f"{key}: Distance = {value['Distance']:.6f}, Weight = {value['Weight']:.6f}")

        # Transform the data using PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_transformed = poly.fit_transform(np.array(X_cumulative))

        # Fit objective and constraint models
        objective_model = LinearRegression()
        constraint_model = LinearRegression()

        objective_model.fit(X_transformed, np.array(y_obj_cumulative), sample_weight=weights)
        constraint_model.fit(X_transformed, np.array(y_const_cumulative), sample_weight=weights)

        # Define surrogate subproblem
        def subproblem_objective(delta):
            delta_transformed = poly.transform([x + delta])
            return objective_model.predict(delta_transformed)[0]

        def subproblem_constraint(delta):
            delta_transformed = poly.transform([x + delta])
            return constraint_model.predict(delta_transformed)[0]

        # Solve subproblem
        res = minimize(subproblem_objective, np.zeros_like(x), method='SLSQP',
                      constraints={'type': 'ineq', 'fun': subproblem_constraint},
                      bounds=[(-radius, radius)] * len(x))
        # FGN: Don't you check if minimization is successful (the return code)?
        delta = res.x

        # Calculate actual-to-predicted ratio (p_k)
        real_constraint_value = constraint(x + delta, mu, A_values)
        predicted_constraint_value = subproblem_constraint(delta)
        real_objective_value = objective(x + delta)
        predicted_objective_value = subproblem_objective(delta)

        px = current_obj + penalty_factor * max(0, -1 * current_cons)
        pxx = objective(x + delta) + penalty_factor * max(0, -1 * constraint(x + delta, mu, A_values))

        mp = subproblem_objective(0) + penalty_factor * max(0, -1 * subproblem_constraint(0))
        mpp = subproblem_objective(delta) + penalty_factor * max(0, -1 * subproblem_constraint(delta))

        pk = (px - pxx) / (mp - mpp) if (mp - mpp) != 0 else 0
        print("px=",px)
        print("mp=",mp)
        print("pxx=",pxx)
        print("mpp=",mpp)
        # Adjust trust region radius
        #if pk <= 0.1:
        #    count += 1
        #if count >= 10:
        #    radius = initial_radius
        #    count = 0
        if pk >= 0.6:
            radius = min(2 * radius, 2.0)
        elif pk < 0.1:
            radius = max(0.5 * radius, 0.001)

        # FGN: Add the new evaluation to the data set	
        X_cumulative.append(x+delta)
        y_obj_cumulative.append(real_objective_value)
        y_const_cumulative.append(real_constraint_value)

        # Update solution if improvement is sufficient
        if pk >= 0.1:
            x = x + delta
            #FGN : also store the evaluation values here so that you do not re-evalute at x 
            current_obj = real_objective_value
            current_cons = real_constraint_value

        # Log progress
        if iteration % 1 == 0:
            print(f"Iteration {iteration}:")
            print(f"Radius: {radius:.6f}")
            print(f"pk: {pk:.6f}")
            print(f"Objective: {objective(x):.6f}")
            print(f"Constraint: {constraint(x, mu, A_values):.6f}")
            print(f"Historical data points: {len(X_cumulative)}")
            print("------------------------")

        # Store values for logging
        constraint_values.append(real_constraint_value)
        predicted_constraint_values.append(predicted_constraint_value)
        objective_values.append(real_objective_value)
        predicted_objective_values.append(predicted_objective_value)

        # Store results
        results.append((iteration, objective(x), x, real_constraint_value, predicted_constraint_value,
                       real_objective_value, predicted_objective_value))

    return results, constraint_values, predicted_constraint_values, objective_values, predicted_objective_values




def save_results(results, filename='optimization_results_polynomial.csv'):
    """Save optimization results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Objective Value'] +
                       [f'x{i+1}' for i in range(n)] +
                       ['Real Constraint Value', 'Predicted Constraint Value',
                        'Real Objective Value', 'Predicted Objective Value'])
        for result in results:
            writer.writerow([result[0], result[1]] +
                          list(result[2]) +
                          [result[3], result[4], result[5], result[6]])

def main():
    """Main function to run the optimization."""
    # Calculate A_values once
    A_values = np.array([A(m) for m in mu])

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Initialize starting point
    x0 = np.random.randn(n)
    print("Initial point:", x0)

    # Run optimization
    results, constraint_values, predicted_constraint_values, objective_values, predicted_objective_values = \
        trust_region_optimization(x0, mu, A_values)

    # Save results
    #save_results(results)
    print("Optimization complete. Results saved to 'optimization_results_polynomial.csv'.")

if __name__ == "__main__":
    main()
