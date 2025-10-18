import numpy as np

# -----------------------------
# 1) Model primitives
# -----------------------------
theta1 = 0.1   # linear cost term
theta2 = 0.05  # quadratic cost term
theta3 = 0.3   # Pr(stay in same x) when keeping
theta4 = 15.0  # replacement cost
beta   = 0.95  # discount factor

K = 90                 # number of states
X = np.arange(K)       # grid: x = 0..89
X_max = K - 1

tol = 1e-8
max_iter = 10000

# Toggle this to use Rust-style logit EV (Type-I EV shocks)
use_logit_ev = False
EULER_GAMMA = 0.5772156649015329  # Euler–Mascheroni constant


def running_cost(x):
    """c(x) = θ1 x + θ2 x^2 (vectorized over x)."""
    return theta1 * x + theta2 * (x**2)


def bellman_update(V):
    """
    One Bellman update returning:
      V_new : updated value function
      keep  : action value for 'keep' at each x
      rep   : action value for 'replace' at each x
    """
    # Expected continuation under 'keep':
    # EV_keep_next(x) = θ3 * V(x) + (1-θ3) * V(min(x+1, X_max))
    V_next = V.copy()
    V_next_shift = V[np.minimum(X + 1, X_max)]
    EV_keep_next = theta3 * V_next + (1 - theta3) * V_next_shift

    keep = -running_cost(X) + beta * EV_keep_next

    # 'replace' resets to x=0 next period
    rep = -theta4 + beta * V[0] * np.ones_like(X)

    if use_logit_ev:
        # Rust EV formulation with type-I EV shocks:
        # EV(x) = γ + log( exp(v_keep(x)) + exp(v_rep(x)) )
        # Here keep/rep are the deterministic parts v_a(x)
        # Do log-sum-exp stably:
        m = np.maximum(keep, rep)
        V_new = EULER_GAMMA + np.log(np.exp(keep - m) + np.exp(rep - m)) + m
    else:
        # Hard-max Bellman
        V_new = np.maximum(keep, rep)

    return V_new, keep, rep


def solve_value_function():
    """Value-function iteration until convergence."""
    V = np.zeros(K)
    for it in range(max_iter):
        V_new, keep, rep = bellman_update(V)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            return V, keep, rep, it + 1, diff
    # If not converged
    return V, keep, rep, max_iter, diff


def extract_policy(V):
    """Given converged V, recompute action values and derive policy & threshold."""
    _, keep, rep = bellman_update(V)  # reuse same logic for action values
    policy = np.where(rep > keep, 1, 0)  # 1=replace, 0=keep
    # threshold = smallest x where replace is optimal (if any)
    repl_idx = np.where(policy == 1)[0]
    x_star = int(repl_idx[0]) if repl_idx.size > 0 else None
    return policy, x_star, keep, rep


if __name__ == "__main__":
    V, keep_last, rep_last, iters, final_diff = solve_value_function()
    policy, x_star, keep_star, rep_star = extract_policy(V)

    print(f"Converged in {iters} iterations (sup-norm diff={final_diff:.2e})")
    print(f"Replacement threshold x*: {x_star}")
    # Show a small table around the threshold
    if x_star is None:
        window = range(0, 8)
    else:
        lo = max(0, x_star - 4)
        hi = min(X_max, x_star + 4)
        window = range(lo, hi + 1)

    print("\n x |     Keep(x)     Rep(x)    Policy")
    print("--------------------------------------")
    for x in window:
        act = "REPLACE" if policy[x] == 1 else "KEEP"
        print(f"{x:2d} | {keep_star[x]:12.6f}  {rep_star[x]:8.6f}   {act}")
