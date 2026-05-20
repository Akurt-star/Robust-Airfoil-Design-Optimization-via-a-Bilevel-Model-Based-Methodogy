import numpy as np
from bayes_opt import BayesianOptimization
import sys, os, shutil
sys.path.append(os.environ["SU2_RUN"])
import SU2
from scipy.stats import norm
from numpy.linalg import matrix_rank
import random
import csv
from datetime import datetime

#Seed also given in main function too.
np.random.seed(42)
random.seed(42)

# ============================================================
# ITERATION LOGGING
# ============================================================
su2_call_counter = 0          # total SU2 simulations
iteration_history = []         # list of dicts, one per outer iteration

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

# Fixed flight conditions for optimization
FIXED_MACH = 0.715
FIXED_AOA = 2.82

def evaluation_f(dv, mach=None, aoa=None):
    global last_design, config, state, last_mach, last_aoa
    global su2_call_counter

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

        su2_call_counter += 1   # <--- COUNT EVERY SU2 CALL
        print(f"[SU2 call #{su2_call_counter}] design: {design.folder}, Mach: {config['MACH_NUMBER']}, AOA: {config['AOA']}")

        factor = 1e-3 * 1e-6
        obj_value = design.obj_f(dv)
        if isinstance(obj_value, (list, tuple, np.ndarray)):
            obj_value = obj_value[0]
        elif not np.isscalar(obj_value):
            raise ValueError(f"[ERROR] Unexpected obj_value format: {type(obj_value)}")
        drag = obj_value / factor

        cons_value = design.con_cieq(dv)
        lift = 0.724 + (-1 * cons_value[0]) / factor
        moment = cons_value[1] / factor + 0.093
        thickness = 0.12 + (-1 * cons_value[2]) / factor

        print("evaluated x=",dv)
        print("lift=", lift)
        print("drag=", drag)
        print("corresponding -lift/drag(objective)=", -1*(lift/drag))
        print("corresponding thickness=", thickness)
        print("corresponding moment=", moment)

        return drag, lift, moment, thickness

    except Exception as e:
        print(f"[ERROR] SU2 evaluation failed at dv={dv}, mach={mach}, aoa={aoa}")
        print(f"[EXCEPTION] {e}")
        return None, None, None, None

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

def dragliftetal(x, mach=FIXED_MACH, aoa=FIXED_AOA):
    global last_dv, last_obj, last_con, last_mach, last_aoa

    if isinstance(x, np.ndarray):
        x = x.reshape(-1)
        x = x.tolist()

    # Check if we can reuse previous evaluation
    if is_same_dv(x, last_dv, 1e-10) and is_same_conditions(mach, aoa, last_mach, last_aoa):
        print("used same obj cons since change of x is under tolerance value")
        return last_obj, last_mach, last_aoa, None, None  # FIX: consistent 5-tuple return
    else:
        print(f"next dv: {x}, Mach: {mach}, AOA: {aoa}")
        last_drag, last_lift, last_moment, last_thickness = evaluation_f(x, mach, aoa)
        last_dv = x
        if mach is not None:
            last_mach = mach
        if aoa is not None:
            last_aoa = aoa
        last_obj = -1* (last_lift / last_drag)
        return last_obj, last_mach, last_aoa, last_moment, last_thickness

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

def generate_data(x, radius, n_samples, mach_nominal, aoa_nominal):
    n = len(x)
    X = []

    x = np.array(x)

    for _ in range(n_samples):
        delta = np.zeros(n)
        for i in range(n):
            lower_bound = max(-radius, xb_low[i] - x[i])
            upper_bound = min(radius, xb_up[i] - x[i])
            delta[i] = np.random.uniform(lower_bound, upper_bound)

        x_new = x + delta
        X.append(x_new)
        print("X_new = ",x_new)
        print("returned X = ",np.array(X))

        return np.array(X)


import sympy as sp

def check_global_convex_quadratic(model, poly, X_sample):
    n_features = X_sample.shape[1]
    x_symbols = sp.symbols(f'x0:{n_features}')

    feature_names = poly.get_feature_names_out([f'x{i}' for i in range(n_features)])

    mapping = {}
    for name in feature_names:
        if name == '1':
            mapping[name] = 1
        else:
            term = 1
            for factor in name.split(' '):
                if '^' in factor:
                    var, power = factor.split('^')
                    idx = int(var[1:])
                    term *= x_symbols[idx]**int(power)
                else:
                    idx = int(factor[1:])
                    term *= x_symbols[idx]
            mapping[name] = term

    expr = model.intercept_
    for coef, name in zip(model.coef_, feature_names):
        expr += coef * mapping[name]

    H = sp.hessian(expr, x_symbols)
    eigvals = H.eigenvals()
    is_convex = all(ev >= 0 for ev in eigvals.keys())

    return is_convex, H, eigvals


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

def trust_region_robust(x0, radius_init, n_iter, mach_nominal, aoa_nominal, n_samples=1):
    global su2_call_counter, iteration_history

    x = np.array(x0)
    x_initial = x.copy()
    radius = radius_init

    # Data to fit surrogate
    X_cum = []
    y_obj_cum = []
    y_thickness_cum = []

    # Evaluate at initial point
    su2_before_init = su2_call_counter
    obj_val_current, _, _, _, thickness_val = dragliftetal(x.tolist())
    X_cum.append(x)
    y_obj_cum.append(obj_val_current)
    y_thickness_cum.append(thickness_val)

    # Log initialization
    iteration_history.append({
        "iteration": 0,
        "phase": "init_center",
        "objective_CL_CD": -1 * obj_val_current if obj_val_current else None,
        "thickness": thickness_val,
        "trust_region_radius": radius,
        "pk": None,
        "step_accepted": None,
        "model_degree": None,
        "delta_norm": None,
        "su2_calls_this_iter": su2_call_counter - su2_before_init,
        "su2_calls_cumulative": su2_call_counter,
        "mach": FIXED_MACH,
        "aoa": FIXED_AOA,
    })

    n_samples_copy = n_samples
    su2_before_sampling = su2_call_counter

    # Generate new samples around current x
    while n_samples_copy > 0:
      X_new1 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

      max_distance = 10.0*radius
      X_array_cum = np.array(X_cum)
      distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
      within_radius_mask_cum = distances_cum <= max_distance
      X_filtered_cum = X_array_cum[within_radius_mask_cum]

      X_cum_new = X_cum.copy()
      X_cum_new.extend(X_new1)

      X_array_cum_new = np.array(X_cum_new)
      distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
      within_radius_mask_cum_new = distances_cum_new <= max_distance
      X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

      poly2 = PolynomialFeatures(degree=2)
      X_trans_cum = poly2.fit_transform(np.array(X_filtered_cum))
      X_trans_cum_new = poly2.fit_transform(np.array(X_filtered_cum_new))

      if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
        print("Point is linearly independent, evaluation will start")
        obj_val1, worst_mach1, worst_aoa1, moment_val1, thickness_val1 = dragliftetal(X_new1[0].tolist())

        X_cum = X_cum_new
        y_obj_cum.append(obj_val1)
        print("obj_val_from_findworst",obj_val1)
        print("y_obj_cum = ",y_obj_cum)
        y_thickness_cum.append(thickness_val1)
        n_samples_copy -= 1
        print("y_thickness_cum = ",y_thickness_cum)
      else:
        print("Point is linearly dependent, point is rejected.")

    # Log initial sampling phase
    iteration_history.append({
        "iteration": 0,
        "phase": "init_sampling",
        "objective_CL_CD": None,
        "thickness": None,
        "trust_region_radius": radius,
        "pk": None,
        "step_accepted": None,
        "model_degree": None,
        "delta_norm": None,
        "su2_calls_this_iter": su2_call_counter - su2_before_sampling,
        "su2_calls_cumulative": su2_call_counter,
        "mach": FIXED_MACH,
        "aoa": FIXED_AOA,
    })

    for k in range(n_iter):

        print(f"\n--- Iteration {k+1}/{n_iter} ---")
        su2_before_iter = su2_call_counter
        model_degree_used = 2  # track which model degree was ultimately used

        num_data = 1
        while num_data > 0:
          X_new2 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

          max_distance = 10.0*radius
          X_array_cum = np.array(X_cum)
          distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
          within_radius_mask_cum = distances_cum <= max_distance
          X_filtered_cum = X_array_cum[within_radius_mask_cum]

          X_cum_new = X_cum.copy()
          X_cum_new.extend(X_new2)

          X_array_cum_new = np.array(X_cum_new)
          distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
          within_radius_mask_cum_new = distances_cum_new <= max_distance
          X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

          poly2 = PolynomialFeatures(degree=2)
          X_trans_cum = poly2.fit_transform(np.array(X_filtered_cum))
          X_trans_cum_new = poly2.fit_transform(np.array(X_filtered_cum_new))

          if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
              print("Point is linearly independent, evaluation will start")
              obj_val2, worst_mach2, worst_aoa2, moment_val2, thickness_val2 = dragliftetal(X_new2[0].tolist())

              X_cum = X_cum_new
              y_obj_cum.append(obj_val2)
              y_thickness_cum.append(thickness_val2)
              num_data -= 1
          else:
              print("Point is linearly dependent, point is rejected.")

        # Fit surrogate models (2nd degree poly regression)
        max_distance = 10.0*radius
        X_array = np.array(X_cum)
        distances = np.max(np.abs(X_array - x), axis=1)
        within_radius_mask = distances <= max_distance

        X_filtered = X_array[within_radius_mask]
        y_obj_filtered = np.array(y_obj_cum)[within_radius_mask]
        y_thickness_filtered = np.array(y_thickness_cum)[within_radius_mask]

        # Keep a copy for degree-1 fallback RMSE
        y_obj_filtered2 = y_obj_filtered.copy()

        print(f"Vekil model için toplam {len(X_cum)} noktadan {len(X_filtered)} tanesi kullanılıyor.")

        poly2 = PolynomialFeatures(degree=2)
        X_trans2 = poly2.fit_transform(np.array(X_filtered))

        obj_model2 = LinearRegression()
        obj_model2.fit(X_trans2, np.array(y_obj_filtered))

        thickness_model2 = LinearRegression()
        thickness_model2.fit(X_trans2, np.array(y_thickness_filtered))

        is_convex1, H_matrix1, eigvals1 = check_global_convex_quadratic(obj_model2, poly2, X_filtered)
        print(f"Objective model globally convex? {is_convex1}")

        is_convex2, H_matrix2, eigvals2 = check_global_convex_quadratic(thickness_model2, poly2, X_filtered)
        print(f"Thickness model globally convex? {is_convex2}")

        y_pred2 = obj_model2.predict(X_trans2)
        rmse2 = np.sqrt(mean_squared_error(y_obj_filtered, y_pred2))
        print(f"[Degree 2] RMSE: {rmse2:.6e}")

        # Subproblem definition
        def surrogate_obj(delta):
            delta = np.array(delta)
            x_trial = x + delta
            x_trial = np.clip(x_trial, xb_low, xb_up)
            return obj_model2.predict(poly2.transform([x_trial]))[0]

        def surrogate_thickness(delta):
            delta = np.array(delta)
            x_trial = x + delta
            x_trial = np.clip(x_trial, xb_low, xb_up)
            return thickness_model2.predict(poly2.transform([x_trial]))[0]

        bounds = []
        for i in range(len(x)):
            lower = max(-radius, xb_low[i] - x[i])
            upper = min(radius, xb_up[i] - x[i])
            bounds.append((lower, upper))

        constraints = [{
            'type': 'ineq',
            'fun': lambda d: -0.12 + surrogate_thickness(d)
        }]

        res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2, "disp":True})
        delta = res.x
        print("degree2 model's solver succeed",res.success)

        if delta is not None and np.all(delta == 0):
            print("Degree 2 modelden delta 0 geldi")
            model_degree_used = 1
            poly1 = PolynomialFeatures(degree=1)
            X_trans1 = poly1.fit_transform(np.array(X_filtered))

            obj_model1 = LinearRegression()
            obj_model1.fit(X_trans1, np.array(y_obj_filtered))

            thickness_model1 = LinearRegression()
            thickness_model1.fit(X_trans1, np.array(y_thickness_filtered))

            is_convex3, H_matrix3, eigvals3 = check_global_convex_quadratic(obj_model1, poly1, X_filtered)
            print(f"Objective model globally convex? {is_convex3}")

            is_convex4, H_matrix4, eigvals4 = check_global_convex_quadratic(thickness_model1, poly1, X_filtered)
            print(f"Thickness model globally convex? {is_convex4}")

            y_pred1 = obj_model1.predict(X_trans1)
            rmse1 = np.sqrt(mean_squared_error(y_obj_filtered, y_pred1))
            print(f"[Degree 1] RMSE: {rmse1:.6e}")

            def surrogate_obj(delta):
                delta = np.array(delta)
                x_trial = x + delta
                x_trial = np.clip(x_trial, xb_low, xb_up)
                return obj_model1.predict(poly1.transform([x_trial]))[0]

            def surrogate_thickness(delta):
                delta = np.array(delta)
                x_trial = x + delta
                x_trial = np.clip(x_trial, xb_low, xb_up)
                return thickness_model1.predict(poly1.transform([x_trial]))[0]

            bounds = []
            for i in range(len(x)):
                lower = max(-radius, xb_low[i] - x[i])
                upper = min(radius, xb_up[i] - x[i])
                bounds.append((lower, upper))

            constraints = [{
                'type': 'ineq',
                'fun': lambda d: -0.12 + surrogate_thickness(d)
            }]

            res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2})
            delta = res.x
            print(f"delta = {delta}")
            print("degree1 model's solver succeed or not=",res.success)

            if delta is not None and np.all(delta == 0):
                for i in range(1):
                    print(f"In Degree 1 delta is 0, we generate new data in smaller radius = {radius}, attempt number {i+1}")
                    num_data_last = 1
                    radius = 0.5*radius
                    while num_data_last > 0:
                      X_new3 = generate_data(x, radius, 1, mach_nominal, aoa_nominal)

                      max_distance = 10.0*radius
                      X_array_cum = np.array(X_cum)
                      distances_cum = np.max(np.abs(X_array_cum - x), axis=1)
                      within_radius_mask_cum = distances_cum <= max_distance
                      X_filtered_cum = X_array_cum[within_radius_mask_cum]

                      X_cum_new = X_cum.copy()
                      X_cum_new.extend(X_new3)

                      X_array_cum_new = np.array(X_cum_new)
                      distances_cum_new = np.max(np.abs(X_array_cum_new - x), axis=1)
                      within_radius_mask_cum_new = distances_cum_new <= max_distance
                      X_filtered_cum_new = X_array_cum_new[within_radius_mask_cum_new]

                      poly1 = PolynomialFeatures(degree=1)
                      X_trans_cum = poly1.fit_transform(np.array(X_filtered_cum))
                      X_trans_cum_new = poly1.fit_transform(np.array(X_filtered_cum_new))

                      if matrix_rank(np.array(X_trans_cum_new)) > matrix_rank(np.array(X_trans_cum)):
                        print("Point is linearly independent, evaluation will start3")
                        obj_val3, worst_mach3, worst_aoa3, moment_val3, thickness_val3 = dragliftetal(X_new3[0].tolist())

                        X_cum = X_cum_new
                        y_obj_cum.extend([obj_val3])
                        y_thickness_cum.extend([thickness_val3])
                        num_data_last -= 1
                      else:
                        print("Point is linearly dependent, point is rejected.")

                    max_distance = 10.0*radius
                    X_array = np.array(X_cum)
                    distances = np.max(np.abs(X_array - x), axis=1)
                    within_radius_mask = distances <= max_distance

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

                        is_convex4, H_matrix4, eigvals4 = check_global_convex_quadratic(thickness_model1, poly1, X_filtered3)
                        print(f"Thickness model globally convex? {is_convex4}")

                        y_pred1 = obj_model1.predict(X_trans1)
                        rmse1 = np.sqrt(mean_squared_error(y_obj_filtered2, y_pred1))
                        print(f"[Degree 1] çok kötüyüz RMSE: {rmse1:.6e}")

                        def surrogate_obj(delta):
                            delta = np.array(delta)
                            x_trial = x + delta
                            x_trial = np.clip(x_trial, xb_low, xb_up)
                            return obj_model1.predict(poly1.transform([x_trial]))[0]

                        def surrogate_thickness(delta):
                            delta = np.array(delta)
                            x_trial = x + delta
                            x_trial = np.clip(x_trial, xb_low, xb_up)
                            return thickness_model1.predict(poly1.transform([x_trial]))[0]

                        bounds = []
                        for i in range(len(x)):
                            lower = max(-radius, xb_low[i] - x[i])
                            upper = min(radius, xb_up[i] - x[i])
                            bounds.append((lower, upper))

                        constraints = [{
                            'type': 'ineq',
                            'fun': lambda d: -0.12 + surrogate_thickness(d)
                        }]

                        res = minimize(surrogate_obj, np.zeros_like(x), method='trust-constr', bounds=bounds, constraints=constraints, options={"verbose":2, "disp":True})
                        delta = res.x
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
        real_obj_val, _, _, moment_val, thickness_val_new = dragliftetal(x_new.tolist())
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

        # Update radius
        radius_before = radius
        if pk >= 0.6:
            radius = min(2 * radius, 2e-5)
        elif pk < 0.01:
            radius = max(0.5 * radius, 1e-7)

        # Accept step?
        step_accepted = pk >= 0.01
        if step_accepted:
            x = x_new
            obj_val_current = real_obj_val
            thickness_val = thickness_val_new
            print("✔ Step accepted.")
        else:
            print("✘ Step rejected.")

        print(f"New radius: {radius:.4e}")
        print(f"Current design: {x}")

        # ============================================================
        # LOG THIS ITERATION
        # ============================================================
        iteration_history.append({
            "iteration": k + 1,
            "phase": "outer",
            "objective_CL_CD": -1 * obj_val_current if obj_val_current else None,
            "thickness": thickness_val,
            "trust_region_radius": radius,
            "pk": pk,
            "step_accepted": step_accepted,
            "model_degree": model_degree_used,
            "delta_norm": float(np.linalg.norm(delta)),
            "su2_calls_this_iter": su2_call_counter - su2_before_iter,
            "su2_calls_cumulative": su2_call_counter,
            "mach": FIXED_MACH,
            "aoa": FIXED_AOA,
        })

    return x, obj_val_current


# ============================================================
# PRINTING AND EXPORTING THE ITERATION TABLE
# ============================================================
def print_iteration_table(history):
    """Print a formatted iteration table to stdout."""
    print("\n" + "="*130)
    print("DETERMINISTIC OPTIMIZATION — ITERATION SUMMARY")
    print(f"Fixed conditions: Mach = {FIXED_MACH}, AoA = {FIXED_AOA}")
    print("="*130)

    header = f"{'Iter':>4} {'Phase':<14} {'CL/CD':>10} {'Thickness':>10} {'TR Radius':>12} {'pk':>10} {'Accepted':>8} {'Deg':>4} {'||δ||':>12} {'SU2(iter)':>10} {'SU2(cum)':>10}"
    print(header)
    print("-"*130)

    for row in history:
        it = row["iteration"]
        phase = row["phase"]
        clcd = f"{row['objective_CL_CD']:.4f}" if row["objective_CL_CD"] is not None else "—"
        thick = f"{row['thickness']:.6f}" if row["thickness"] is not None else "—"
        tr = f"{row['trust_region_radius']:.4e}"
        pk_val = f"{row['pk']:.4e}" if row["pk"] is not None else "—"
        acc = "Yes" if row["step_accepted"] == True else ("No" if row["step_accepted"] == False else "—")
        deg = str(row["model_degree"]) if row["model_degree"] is not None else "—"
        dn = f"{row['delta_norm']:.4e}" if row["delta_norm"] is not None else "—"
        su2_it = str(row["su2_calls_this_iter"])
        su2_cum = str(row["su2_calls_cumulative"])

        print(f"{it:>4} {phase:<14} {clcd:>10} {thick:>10} {tr:>12} {pk_val:>10} {acc:>8} {deg:>4} {dn:>12} {su2_it:>10} {su2_cum:>10}")

    print("="*130)
    print(f"TOTAL SU2 SIMULATIONS: {su2_call_counter}")
    print("="*130)


def export_iteration_csv(history, filename="deterministic_iterations.csv"):
    """Export iteration history to CSV."""
    if not history:
        return

    keys = history[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"Iteration history exported to {filename}")


def main():

    np.random.seed(42)
    # Nominal flight conditions
    mach_nominal = FIXED_MACH
    aoa_nominal = FIXED_AOA

    # Initial design
    x0 = dv0
    radius_init = 0.000005
    n_iter = 10
    n_samples = 10

    print(">>> Trust Region DETERMINISTIC Optimization <<<")
    print(f"Fixed Mach: {FIXED_MACH}, Fixed AoA: {FIXED_AOA}")
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
    print(f"Final deterministic objective (CL/CD): {-1 * obj_best:.6e}")

    # Print and export the iteration table
    print_iteration_table(iteration_history)
    export_iteration_csv(iteration_history)

if __name__ == "__main__":
    main()
