import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import csv
from scipy import linalg as LA

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

def generate_data(x, radius, mu, A_values, n_samples=30):
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

def trust_region_optimization(x0, mu, A_values, max_iterations=100, initial_radius=1.0):
    """Main trust region optimization function."""
    x = x0
    radius = initial_radius
    results = []
    objective_values = []
    predicted_objective_values = []
    constraint_values = []
    predicted_constraint_values = []

    for iteration in range(max_iterations):
        # Generate data points and fit models
        X, y_obj, y_const = generate_data(x, radius, mu, A_values)

        # Create and fit separate models for objective and constraint
        objective_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=True),
            LinearRegression()
        )
        constraint_model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=True),
            LinearRegression()
        )

        objective_model.fit(X, y_obj)
        constraint_model.fit(X, y_const)

        def subproblem_objective(delta):
            """Surrogate objective function for the subproblem."""
            delta_reshaped = np.array([x + delta])
            return objective_model.predict(delta_reshaped)[0]

        def subproblem_constraint(delta):
            """Surrogate constraint function for the subproblem."""
            delta_reshaped = np.array([x + delta])
            return constraint_model.predict(delta_reshaped)[0]

        # Solve the trust region subproblem
        res = minimize(subproblem_objective, np.zeros(n), method='SLSQP',
                      constraints={'type': 'ineq', 'fun': subproblem_constraint},
                      bounds=[(-radius, radius)] * n,
                      options={'ftol': 1e-8, 'maxiter': 200})

        delta = res.x
        print("Status:", res.status)
        # Calculate actual and predicted values
        real_constraint_value = constraint(x + delta, mu, A_values)
        predicted_constraint_value = subproblem_constraint(delta)
        real_objective_value = objective(x + delta)
        predicted_objective_value = subproblem_objective(delta)

        # Store values for logging
        constraint_values.append(real_constraint_value)
        predicted_constraint_values.append(predicted_constraint_value)
        objective_values.append(real_objective_value)
        predicted_objective_values.append(predicted_objective_value)

        # Handle very small changes
        px = objective(x) + 1000 * max(0, -1*constraint(x, mu, A_values))
        pxx = objective(x + delta) + 1000 * max(0, -1*constraint(x + delta, mu, A_values))

        mp = subproblem_objective(0) + 1000 * max(0, -1*subproblem_constraint(0))
        mpp = subproblem_objective(delta) + 1000 * max(0, -1*subproblem_constraint(delta))


        pk = (px - pxx) / (mp - mpp) if (mp - mpp) != 0 else 0

        # Update trust region radius
        if pk > 0.9:
            radius = min(2*radius, 2)
        elif pk < 0.1:
            radius = max(0.5*radius, 0.001)

        # Update solution if improvement is sufficient
        if pk > 0.1:
            x = x + delta

        # Log progress
        if iteration % 5 == 0:
            print(f"Iteration {iteration}:")
            print(f"Radius: {radius:.6f}")
            print(f"pk: {pk:.6f}")
            print(f"Objective: {real_objective_value:.6f}")
            print(f"Constraint: {real_constraint_value:.6f}")
            print("------------------------")

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
    #np.random.seed(4)

    # Initialize starting point
    x0 = np.random.randn(n)
    print("Initial point:", x0)

    # Run optimization
    results, constraint_values, predicted_constraint_values, objective_values, predicted_objective_values = \
        trust_region_optimization(x0, mu, A_values)

    # Save results
    save_results(results)
    print("Optimization complete. Results saved to 'optimization_results_polynomial.csv'.")

if __name__ == "__main__":
    main()