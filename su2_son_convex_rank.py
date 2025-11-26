import numpy as np
from bayes_opt import BayesianOptimization
import sys, os, shutil
sys.path.append(os.environ["SU2_RUN"])
import SU2
from scipy.stats import norm
from numpy.linalg import matrix_rank
import random

#Seed also given in main function too.
np.random.seed(42)
random.seed(42)

# Global variables
last_dv = None  # en son func eval yapilan dv
last_obj = None
last_con = None
last_design = None
last_mach = None
last_aoa = None

# TOLERANSLAR VS BELİRLENDİİ
TOL_DELTA = 1e-8
TOL_XCHANGE = 1e-8

# SU2 configuration
config = SU2.io.Config("turb_SA_RAE2822_python.cfg")
config["HISTORY_OUTPUT"].append("AERO_COEFF")
config.NZONES = 1  # no multizone simulation
config.NUMBER_PART = 1
config.CONSOLE = "CONCISE"
config.GRADIENT_METHOD = "NONE"

# Current Mach and AOA initialization
mach_curr = config["MACH_NUMBER"]
aoa_curr = config["AOA"]  # Initial AOA value from config

def evaluation_f(dv, mach=None, aoa=None):
    import numpy as np  # emin ol localde de np erişilebilir
    global last_design, config, state, last_mach, last_aoa

    try:
        if mach is not None:
            config["MACH_NUMBER"] = mach
            last_mach = mach
        if aoa is not None:
            config["AOA"] = aoa
            last_aoa = aoa

        config.unpack_dvs(dv)
        state = SU2.io.State()
        state.find_files(config)
        design = SU2.eval.Design(config, state)
        last_design = design
        print(f"evaluation at design: {design.folder}, Mach: {config['MACH_NUMBER']}, AOA: {config['AOA']}")

        obj_value = design.obj_f(dv)
        cons_value = design.con_cieq(dv)  # con_cieq assumes the form c(x)<=0

        return obj_value[0], cons_value

    except Exception as e:
        print(f"[ERROR] SU2 evaluation failed at dv={dv}, mach={mach}, aoa={aoa}")
        print(f"[EXCEPTION] {e}")
        return None, [None, None, None]


def is_same_dv(dv1, dv2, tol):
    if dv1 is None or dv2 is None:
        return False
    dv1 = np.array(dv1)
    dv2 = np.array(dv2)
    return np.linalg.norm(dv1 - dv2) < tol

def is_same_conditions(mach1, aoa1, mach2, aoa2, tol=1e-6):
    if mach1 is None or aoa1 is None or mach2 is None or aoa2 is None:
        return False
    return (abs(mach1 - mach2) < tol) and (abs(aoa1 - aoa2) < tol)

def dragliftetal(x, mach=None, aoa=None):
    global last_dv, last_obj, last_con, last_mach, last_aoa

    if isinstance(x, np.ndarray):
        x = x.reshape(-1)
        x = x.tolist()

    # Check if we can reuse previous evaluation
    if is_same_dv(x, last_dv, 1e-10) and is_same_conditions(mach, aoa, last_mach, last_aoa):
        return last_obj, np.array(last_con)
    else:
        print(f"next dv: {x}, Mach: {mach}, AOA: {aoa}")
        last_obj, last_con = evaluation_f(x, mach, aoa)
        last_dv = x
        if mach is not None:
            last_mach = mach
        if aoa is not None:
            last_aoa = aoa
        return last_obj, np.array(last_con)

# Design variable setup
def_dv = config.DEFINITION_DV  # complete definition of the design variable
n_dv = sum(def_dv["SIZE"])
dv0 = [0.0] * n_dv
relax_factor = float(config.OPT_RELAX_FACTOR)  # line search scale
bound_upper = float(config.OPT_BOUND_UPPER)
bound_lower = float(config.OPT_BOUND_LOWER)
xb_low = [float(bound_lower) / float(relax_factor)] * n_dv  # lower dv bound
xb_up = [float(bound_upper) / float(relax_factor)] * n_dv  # upper dv bound
xb = list(zip(xb_low, xb_up))
print("Design variable bounds set to:", xb)

def find_worst_case_objective(x, mach_nominal, aoa_nominal, uncertain_range=0.1):
    """
        Uses Bayesian optimization to find the worst-case flight conditions (Mach, AOA)
        within the uncertainty range.

        Args:
            x: Design variables
            mach_nominal: Nominal Mach number
            aoa_nominal: Nominal angle of attack
            uncertain_range: Range of uncertainty as a percentage (0.1 = 10%)

        Returns:
            Tuple of (worst_case_mach, worst_case_aoa)
    """
    # Define bounds for uncertain parameters
    pbounds = {
        'mach': (mach_nominal - uncertain_range/10, mach_nominal + uncertain_range/10),
        'aoa': (aoa_nominal - uncertain_range, aoa_nominal + uncertain_range)
    }
    print(f"Starting Bayesian optimization to find worst-case conditions")
    print(f"Mach range: {pbounds['mach']}")
    print(f"AOA range: {pbounds['aoa']}")

    result_log = {}

    # Wrapper function for Bayesian optimization
    # This fixes the design variables and optimizes over uncertain parameters
    def objective_wrapper(mach, aoa):
        # Call SU2 evaluation
        drag, constraints = dragliftetal(x, mach=mach, aoa=aoa)

        if drag == None:
            return -999999
        else:
            factor = 1e-3 * 1e-6

            drag = drag / factor
            lift = 0.724 + (-1 * constraints[0])/factor
            moment_val = constraints[1]/factor + 0.093
            thickness_val = 0.12 + (- 1 * constraints[2])/factor

            if abs(lift) < 1e-6:  # Avoid division by zero
                lift = 1e-6

            # Calculate lift / drag
            icl_cd = -1 * (lift / drag)

            key = (mach, aoa)
            result_log[key] = {
                "drag": drag,
                "lift": lift,
                "icl_cd": -1 * icl_cd,
                "moment": moment_val,
                "thickness": thickness_val
            }

            print(f"Evaluated at Mach={mach:.4e}, AOA={aoa:.4e}: Drag={drag:.6e}, Lift={lift:.6e}, Cl/Cd={-1 * icl_cd:.6e}")

            # Return ICl/Cd as the objective to minimize
            return icl_cd

    # Initialize Bayesian optimization
    optimizer = BayesianOptimization(
        f=objective_wrapper,
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )

    # Run one iteration of Bayesian optimization
    # 3 initial random points, 1 optimization step
    optimizer.maximize(init_points=2, n_iter=5)

    # Get the worst-case conditions
    worst_case = optimizer.max
    worst_case_mach = worst_case['params']['mach']
    worst_case_aoa = worst_case['params']['aoa']

    key = (worst_case_mach, worst_case_aoa)
    res = result_log.get(key, {})
    moment_val = res.get("moment", None)
    thickness_val = res.get("thickness", None)
    if worst_case["target"] == -999999:
        return None, None, None, None, None
    else:
        print("\nBayesian optimization results:")
        print(f"Worst-case conditions: Mach={worst_case_mach:.4e}, AOA={worst_case_aoa:.4e}")
        print(f"Cl/Cd at these conditions: {worst_case['target']:.6e}")
        print(f"Moment value at these conditions= {moment_val}, thickness value at these condition={thickness_val}")

        return worst_case["target"], worst_case_mach, worst_case_aoa, moment_val, thickness_val

def generate_data(x, radius, n_samples, mach_nominal, aoa_nominal):
    """
    Generate data samples around a design point x by perturbing it within a radius,
    respecting the variable bounds.

    Args:
        x: Initial design variable vector.
        radius: Perturbation radius for the design variables.
        n_samples: Number of data points to generate.
        mach_nominal: Nominal Mach number for worst-case search.
        aoa_nominal: Nominal angle of attack for worst-case search.

    Returns:
        X: Array of design variables.
        y_obj: Array of worst-case objectives.
        y_const: Array of constraint values (e.g., thickness).
    """
    n = len(x)
    X = []
    y_obj = []
    y_const = []

    x = np.array(x)

    for _ in range(n_samples):
        # Generate perturbation within bounds
        delta = np.zeros(n)
        for i in range(n):
            # Calculate effective bounds for this variable
            lower_bound = max(-radius, xb_low[i] - x[i])  # max{-radius_i, l-x^i_k}
            upper_bound = min(radius, xb_up[i] - x[i])    # min{radius_i, u^i-x^i_k}

            # Sample from the effective bounds
            delta[i] = np.random.uniform(lower_bound, upper_bound)

        x_new = x + delta
        X.append(x_new)
        print("X_new = ",x_new)
        print("returned X = ",np.array(X))

        return np.array(X)


import sympy as sp

def check_global_convex_quadratic(model, poly, X_sample):
    """
    PolynomialRegression modelinin (degree=2) global convexliğini kontrol eder.
    X_sample: poly.fit_transform edilmeden önceki örnek veri (ör. X_selected)
    """
    n_features = X_sample.shape[1]  # giriş boyutu
    x_symbols = sp.symbols(f'x0:{n_features}')  # x0, x1, ..., x_{n-1}

    feature_names = poly.get_feature_names_out([f'x{i}' for i in range(n_features)])

    # mapping'i otomatik oluştur
    mapping = {}
    for name in feature_names:
        if name == '1':
            mapping[name] = 1
        else:
            term = 1
            for factor in name.split(' '):  # 'x0^2', 'x0 x1', ...
                if '^' in factor:
                    var, power = factor.split('^')
                    idx = int(var[1:])
                    term *= x_symbols[idx]**int(power)
                else:
                    idx = int(factor[1:])
                    term *= x_symbols[idx]
            mapping[name] = term

    # Symbolic polinom
    expr = model.intercept_
    for coef, name in zip(model.coef_, feature_names):
        expr += coef * mapping[name]

    # Hessian ve convexlik testi
    H = sp.hessian(expr, x_symbols)
    eigvals = H.eigenvals()
    is_convex = all(ev >= 0 for ev in eigvals.keys())

    return is_convex, H, eigvals


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

def trust_region_robust(x0, radius_init, n_iter, mach_nominal, aoa_nominal, n_samples=1):
    """
    Trust region optimization using robust objective and constraint (thickness)
    from worst-case Bayesian optimization.
    """
    x = np.array(x0)
    x_initial = x.copy()
    radius = radius_init

    # Data to fit surrogate
    X_cum = []
    y_obj_cum = []
    y_thickness_cum = []

    # Evaluate at initial point
    obj_val_current, _, _, _, thickness_val = find_worst_case_objective(x.tolist(), mach_nominal, aoa_nominal)
    X_cum.append(x)
    y_obj_cum.append(obj_val_current)
    y_thickness_cum.append(thickness_val)
    n_samples_copy = n_samples
    # Generate new samples around current x
    while n_samples_copy > 0:
      # Yeni 1 örnek üret
      X_new1 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

      max_distance = 10.0*radius
      X_array_cum = np.array(X_cum)
      distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
      within_radius_mask_cum = distances_cum <= max_distance

      # Apply mask to select data within radius
      X_filtered_cum = X_array_cum[within_radius_mask_cum]

      # X_cum'un bir kopyasını al (bağımsız bir liste)
      X_cum_new = X_cum.copy()
      X_cum_new.extend(X_new1)

      X_array_cum_new = np.array(X_cum_new)
      distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
      within_radius_mask_cum_new = distances_cum_new <= max_distance

      # Apply mask to select data within radius
      X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

      # PolynomialFeatures objesi (tüm veri için aynı kalabilir)
      poly2 = PolynomialFeatures(degree=2)

      # Eski ve yeni genişletilmiş (polynomial) matrisleri hesapla
      X_trans_cum = poly2.fit_transform(np.array(X_filtered_cum))
      X_trans_cum_new = poly2.fit_transform(np.array(X_filtered_cum_new))

      # Rank kontrolü: Eğer yeni nokta rank'ı artırıyorsa, kabul et
      if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
        print("Point is linearly independent, evaluation will start")
        obj_val1, worst_mach1, worst_aoa1, moment_val1, thickness_val1 = find_worst_case_objective(X_new1[0].tolist(), mach_nominal, aoa_nominal)

        X_cum = X_cum_new
        y_obj_cum.append(obj_val1)
        print("obj_val_from_findworst",obj_val1)
        print("y_obj_cum = ",y_obj_cum)
        y_thickness_cum.append(thickness_val1)
        n_samples_copy -= 1
        print("y_thickness_cum = ",y_thickness_cum)
      else:
        print("Point is linearly dependent, point is rejected.")

    for k in range(n_iter):

        print(f"\n--- Iteration {k+1}/{n_iter} ---")
        num_data = 1
        while num_data > 0:
          # Yeni 1 örnek üret
          X_new2 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

          max_distance = 10.0*radius
          X_array_cum = np.array(X_cum)
          distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
          within_radius_mask_cum = distances_cum <= max_distance

          # Apply mask to select data within radius
          X_filtered_cum = X_array_cum[within_radius_mask_cum]

          # X_cum'un bir kopyasını al (bağımsız bir liste)
          X_cum_new = X_cum.copy()
          X_cum_new.extend(X_new2)

          X_array_cum_new = np.array(X_cum_new)
          distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
          within_radius_mask_cum_new = distances_cum_new <= max_distance

          # Apply mask to select data within radius
          X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

          # PolynomialFeatures objesi (tüm veri için aynı kalabilir)
          poly2 = PolynomialFeatures(degree=2)

          # Eski ve yeni genişletilmiş (polynomial) matrisleri hesapla
          X_trans_cum = poly2.fit_transform(np.array(X_filtered_cum))
          X_trans_cum_new = poly2.fit_transform(np.array(X_filtered_cum_new))

          # Rank kontrolü: Eğer yeni nokta rank'ı artırıyorsa, kabul et
          if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
              print("Point is linearly independent, evaluation will start")
              obj_val2, worst_mach2, worst_aoa2, moment_val2, thickness_val2 = find_worst_case_objective(X_new2[0].tolist(), mach_nominal, aoa_nominal)

              X_cum = X_cum_new
              y_obj_cum.append(obj_val2)
              y_thickness_cum.append(thickness_val2)
              num_data -= 1
          else:
              print("Point is linearly dependent, point is rejected.")

        # Fit surrogate models (2nd degree poly regression)
        # Filter data within radius 10 from current point x
        max_distance = 10.0*radius
        X_array = np.array(X_cum)
        distances = np.max(np.abs(X_array - x), axis=1)
        within_radius_mask = distances <= max_distance

        # Apply mask to select data within radius
        X_filtered = X_array[within_radius_mask]
        y_obj_filtered = np.array(y_obj_cum)[within_radius_mask]
        y_thickness_filtered = np.array(y_thickness_cum)[within_radius_mask]

        print(f"Vekil model için toplam {len(X_cum)} noktadan {len(X_filtered)} tanesi kullanılıyor.")

        poly2 = PolynomialFeatures(degree=2)
        X_trans2 = poly2.fit_transform(np.array(X_filtered))

        obj_model2 = LinearRegression()
        obj_model2.fit(X_trans2, np.array(y_obj_filtered))

        thickness_model2 = LinearRegression()
        thickness_model2.fit(X_trans2, np.array(y_thickness_filtered))

        is_convex1, H_matrix1, eigvals1 = check_global_convex_quadratic(obj_model2, poly2, X_filtered)
        print(f"Objective model globally convex? {is_convex1}")
        print(f"Objective models Eigenvalues of Hessian matrix: {eigvals1}")
        print(f"Objective models Hessian matrix: {H_matrix1}")

        is_convex2, H_matrix2, eigvals2 = check_global_convex_quadratic(thickness_model2, poly2, X_filtered)
        print(f"Thickness model globally convex? {is_convex2}")
        print(f"Thickness models Eigenvalues of Hessian matrix: {eigvals2}")
        print(f"Thickness models Hessian matrix: {H_matrix2}")


        y_pred2 = obj_model2.predict(X_trans2)
        rmse2 = np.sqrt(mean_squared_error(y_obj_filtered, y_pred2))
        print(f"[Degree 2] RMSE: {rmse2:.6e}")

        # Subproblem definition
        def surrogate_obj(delta):
            delta = np.array(delta)
            x_trial = x + delta
            # Ensure we stay within global bounds
            x_trial = np.clip(x_trial, xb_low, xb_up)
            return obj_model2.predict(poly2.transform([x_trial]))[0]

        def surrogate_thickness(delta):
            delta = np.array(delta)
            x_trial = x + delta
            # Ensure we stay within global bounds
            x_trial = np.clip(x_trial, xb_low, xb_up)
            return thickness_model2.predict(poly2.transform([x_trial]))[0]

        bounds = []
        for i in range(len(x)):
            # Calculate effective bounds for this variable
            lower = max(-radius, xb_low[i] - x[i])  # max{-radius_i, l-x^i_k}
            upper = min(radius, xb_up[i] - x[i])    # min{radius_i, u^i-x^i_k}
            bounds.append((lower, upper))

        constraints = [{
            'type': 'ineq',
            'fun': lambda d: -0.12 + surrogate_thickness(d)  # thickness must be >= 0.12
        }]

        res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2, "disp":True})
        delta = res.x #verbose
        print("degree2 model's solver succeed",res.success)

        if delta is not None and np.all(delta == 0):
            # Fit surrogate models (2nd degree poly regression)
            print("Degree 2 modelden delta 0 geldi")
            poly1 = PolynomialFeatures(degree=1)
            X_trans1 = poly1.fit_transform(np.array(X_filtered))

            obj_model1 = LinearRegression()
            obj_model1.fit(X_trans1, np.array(y_obj_filtered))

            thickness_model1 = LinearRegression()
            thickness_model1.fit(X_trans1, np.array(y_thickness_filtered))

            is_convex3, H_matrix3, eigvals3 = check_global_convex_quadratic(obj_model1, poly1, X_filtered)
            print(f"Objective model globally convex? {is_convex3}")
            print(f"Objective models Eigenvalues of Hessian matrix: {eigvals3}")
            print(f"Objective models Hessian matrix: {H_matrix3}")

            is_convex4, H_matrix4, eigvals4 = check_global_convex_quadratic(thickness_model1, poly1, X_filtered)
            print(f"Thickness model globally convex? {is_convex4}")
            print(f"Thickness models Eigenvalues of Hessian matrix: {eigvals4}")
            print(f"Thickness models Hessian matrix: {H_matrix4}")


            y_pred1 = obj_model1.predict(X_trans1)
            rmse1 = np.sqrt(mean_squared_error(y_obj_filtered, y_pred1))
            print(f"[Degree 1] RMSE: {rmse1:.6e}")

            # Subproblem definition
            def surrogate_obj(delta):
                delta = np.array(delta)
                x_trial = x + delta
                # Ensure we stay within global bounds
                x_trial = np.clip(x_trial, xb_low, xb_up)
                return obj_model1.predict(poly1.transform([x_trial]))[0]

            def surrogate_thickness(delta):
                delta = np.array(delta)
                x_trial = x + delta
                # Ensure we stay within global bounds
                x_trial = np.clip(x_trial, xb_low, xb_up)
                return thickness_model1.predict(poly1.transform([x_trial]))[0]

            bounds = []
            for i in range(len(x)):
                # Calculate effective bounds for this variable
                lower = max(-radius, xb_low[i] - x[i])  # max{-radius_i, l-x^i_k}
                upper = min(radius, xb_up[i] - x[i])    # min{radius_i, u^i-x^i_k}
                bounds.append((lower, upper))

            constraints = [{
                'type': 'ineq',
                'fun': lambda d: -0.12 + surrogate_thickness(d)  # thickness must be >= 0.12
            }]

            res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2})
            delta = res.x #verbose
            print(f"delta = {delta}")
            print("degree1 model's solver succeed or not=",res.success)

            if delta is not None and np.all(delta == 0):
                for i in range(1):
                    print(f"In Degree 1 delta is 0, we generate new data in smaller radius = {radius}, attempt number {i+1}")
                    num_data_last = 1
                    radius = 0.5*radius
                    while num_data_last > 0:
                      # Yeni 1 örnek üret
                      X_new3 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

                      max_distance = 10.0*radius
                      X_array_cum = np.array(X_cum)
                      distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
                      within_radius_mask_cum = distances_cum <= max_distance

                      # Apply mask to select data within radius
                      X_filtered_cum = X_array_cum[within_radius_mask_cum]

                      # X_cum'un bir kopyasını al (bağımsız bir liste)
                      X_cum_new = X_cum.copy()
                      X_cum_new.extend(X_new3)

                      X_array_cum_new = np.array(X_cum_new)
                      distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
                      within_radius_mask_cum_new = distances_cum_new <= max_distance

                      # Apply mask to select data within radius
                      X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

                      # PolynomialFeatures objesi (tüm veri için aynı kalabilir)
                      poly1 = PolynomialFeatures(degree=1)

                      # Eski ve yeni genişletilmiş (polynomial) matrisleri hesapla
                      X_trans_cum = poly1.fit_transform(np.array(X_filtered_cum))
                      X_trans_cum_new = poly1.fit_transform(np.array(X_filtered_cum_new))

                      # Rank kontrolü: Eğer yeni nokta rank'ı artırıyorsa, kabul et
                      if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
                        print("Point is linearly independent, evaluation will start3")
                        obj_val3, worst_mach3, worst_aoa3, moment_val3, thickness_val3 = find_worst_case_objective(X_new3[0].tolist(), mach_nominal, aoa_nominal)

                        X_cum = X_cum_new
                        y_obj_cum.extend(obj_val3)
                        y_thickness_cum.extend(thickness_val3)
                        num_data_last -= 1
                      else:
                        print("Point is linearly dependent, point is rejected.")

                    max_distance = 10.0*radius
                    X_array = np.array(X_cum)
                    distances = np.max(np.abs(X_array - x), axis=1)
                    within_radius_mask = distances <= max_distance

                    # Apply mask to select data within radius
                    X_filtered3 = X_array[within_radius_mask]
                    y_obj_filtered3 = np.array(y_obj_cum)[within_radius_mask]
                    y_thickness_filtered3 = np.array(y_thickness_cum)[within_radius_mask]

                    if X_new3[0] is not None:
                        poly1 = PolynomialFeatures(degree=1)
                        X_trans1 = poly1.fit_transform(np.array(X_filtered3))

                        obj_model1 = LinearRegression()
                        obj_model1.fit(X_trans1, np.array(y_obj_filtered3))

                        thickness_model1 = LinearRegression()
                        thickness_model1.fit(X_trans1, np.array(y_thickness_filtered3))

                        is_convex3, H_matrix3, eigvals3 = check_global_convex_quadratic(obj_model1, poly1, X_filtered3)
                        print(f"Objective model globally convex? {is_convex3}")
                        #print(f"Objective models Eigenvalues of Hessian matrix: {eigvals3}")
                        #print(f"Objective models Hessian matrix: {H_matrix3}")

                        is_convex4, H_matrix4, eigvals4 = check_global_convex_quadratic(thickness_model1, poly1, X_filtered3)
                        print(f"Thickness model globally convex? {is_convex4}")
                        #print(f"Thickness models Eigenvalues of Hessian matrix: {eigvals4}")
                        #print(f"Thickness models Hessian matrix: {H_matrix4}")


                        y_pred1 = obj_model1.predict(X_trans1)
                        rmse1 = np.sqrt(mean_squared_error(y_obj_filtered2, y_pred1))
                        print(f"[Degree 1] çok kötüyüz RMSE: {rmse1:.6e}")

                        # Subproblem definition
                        def surrogate_obj(delta):
                            delta = np.array(delta)
                            x_trial = x + delta
                            # Ensure we stay within global bounds
                            x_trial = np.clip(x_trial, xb_low, xb_up)
                            return obj_model1.predict(poly1.transform([x_trial]))[0]

                        def surrogate_thickness(delta):
                            delta = np.array(delta)
                            x_trial = x + delta
                            # Ensure we stay within global bounds
                            x_trial = np.clip(x_trial, xb_low, xb_up)
                            return thickness_model1.predict(poly1.transform([x_trial]))[0]

                        bounds = []
                        for i in range(len(x)):
                            # Calculate effective bounds for this variable
                            lower = max(-radius, xb_low[i] - x[i])  # max{-radius_i, l-x^i_k}
                            upper = min(radius, xb_up[i] - x[i])    # min{radius_i, u^i-x^i_k}
                            bounds.append((lower, upper))

                        constraints = [{
                            'type': 'ineq',
                            'fun': lambda d: -0.12 + surrogate_thickness(d)  # thickness must be >= 0.12
                        }]

                        res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2, "disp":True})
                        delta = res.x #verbose
                        print("degree 1 den de 0 geldi yeni delta bu = {delta}")

                        if not(delta is not None and np.all(delta == 0)):
                            print("Degree 1den 0 geldikten sonra radius küçülterek yeni model ile adım oluşturuldu")
                            break
                        else:
                            print(f"we are in optimal with design = {x}")
                    else:
                        print("Bu denemede yeni veri üretilemedi, atlanıyor.")




        # Evaluate real objective and constraint at proposed point
        x_new = x + delta
        real_obj_val, _, _, moment_val, thickness_val_new = find_worst_case_objective(x_new.tolist(), mach_nominal, aoa_nominal)
        print("thickness =", thickness_val_new)
        print("model previous thickness=",surrogate_thickness(0))
        print("model next point thickness=", surrogate_thickness(delta))
        print("moment value=", moment_val)

        # Trust region model-based improvement
        m0 = surrogate_obj(np.zeros_like(x)) - 10000 * min(0, surrogate_thickness(0) - 0.12)
        mp = surrogate_obj(delta) - 10000 * min(0, surrogate_thickness(delta) - 0.12)
        p0 = obj_val_current - 10000 * min(0, thickness_val - 0.12)
        pp = real_obj_val - 10000 * min(0, thickness_val_new - 0.12)

        pk = (p0 - pp) / (m0 - mp) if (m0 - mp) != 0 else 0

        print(f"Current obj: {obj_val_current:.6e} | New obj: {real_obj_val:.6e}")
        print(f"Predicted new obj: {mp:.6e}")
        print(f"pk = {pk:.4e}")

        """# Calculate termination criteria
        norm_delta = np.linalg.norm(delta)
        rel_x_change = np.linalg.norm(x_new - x) / np.linalg.norm(x_initial)

        print(f"Delta norm: {norm_delta:.2e}, Relative x change: {rel_x_change:.2e}")

        # Check termination conditions
        if norm_delta <= TOL_DELTA:
            print("Termination condition 1 met: ||delta_k|| <= 1e-8")
            break
        if rel_x_change <= TOL_XCHANGE:
            print("Termination condition 2 met: |x(k+1)-x(k)|/x_initial <= 1e-8")
            break
        if k == n_iter - 1:
            print("Termination condition 3 met: Max iterations reached")"""

        # Update radius
        if pk >= 0.6:
            radius = min(2 * radius, 2e-5)
        elif pk < 0.01:
            radius = max(0.5 * radius, 1e-7)

        # Accept step?
        if pk >= 0.01:
            x = x_new
            obj_val_current = real_obj_val
            thickness_val = thickness_val_new
            print("✔ Step accepted.")
        else:
            print("✘ Step rejected.")

        print(f"New radius: {radius:.4e}")
        print(f"Current design: {x}")

    return x, obj_val_current

def main():

    np.random.seed(42)
    # Nominal flight conditions
    mach_nominal = config["MACH_NUMBER"]
    aoa_nominal = config["AOA"]

    # Initial design
    x0 = dv0
    radius_init = 0.000005
    n_iter = 10
    n_samples = 10

    print(">>> Trust Region Robust Optimization <<<")
    print(f"Maximum iterations: {n_iter}")
    print(f"Termination tolerance for delta: {TOL_DELTA}")
    print(f"Termination tolerance for relative x change: {TOL_XCHANGE}")

    # Optimize
    x_best, obj_best = trust_region_robust(
        x0, radius_init, n_iter,
        mach_nominal, aoa_nominal,
        n_samples=n_samples
    )

    print("\n>>> Optimization Completed <<<")
    print(f"Final design: {x_best}")
    print(f"Final robust objective: {-1 * obj_best:.6e}")

if __name__ == "__main__":
    main()
